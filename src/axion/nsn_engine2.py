import numpy as np
import warp as wp
import warp.context as wpctx
import warp.optim.linear as wpol
import warp.sparse as wps
from axion.contact_constraint import contact_constraint_kernel
from axion.dynamics_constraint import unconstrained_dynamics_kernel
from axion.optim.cr import cr_solver_graph_compatible
from axion.utils.add_inplace import add_inplace
from warp.sim import Control
from warp.sim import Integrator
from warp.sim import Model
from warp.sim import State

CONTACT_CONSTRAINT_STABILIZATION = 0.1  # Baumgarte stabilization factor
CONTACT_FB_ALPHA = 0.25  # Fisher-Burmeister scaling factor of the first argument
CONTACT_FB_BETA = 0.25  # Fisher-Burmeister scaling factor of the second argument


def generate_mass_block_indices(num_bodies: int, device: wpctx.Device):
    """
    Generate row and column indices for non-zero elements of the mass matrix
    for unconstrained rigid bodies, with velocity vector [omega, v].
    Each body has a 3x3 inertia tensor (9 non-zero elements) and a 3x3 diagonal
    mass block (3 non-zero elements).

    Args:
        num_bodies (int): Number of bodies in the system.
        device (warp.context.Device): The device on which to create the arrays.

    Returns:
        H_rows (np.array): Row indices of non-zero elements.
        H_cols (np.array): Column indices of non-zero elements.
    """
    H_rows = []
    H_cols = []

    for i in range(num_bodies):  # Loop over each body
        # Base index for the current body (6x6 block)
        base = 6 * i

        # Inertia tensor block (3x3, rows/cols: base to base+2)
        for r in range(3):  # Rows of inertia tensor
            for c in range(3):  # Columns of inertia tensor
                H_rows.append(base + r)
                H_cols.append(base + c)

        # Mass block (3x3 diagonal, rows/cols: base+3 to base+5)
        for d in range(3):  # Diagonal elements
            H_rows.append(base + 3 + d)
            H_cols.append(base + 3 + d)
    H_rows = wp.array(H_rows, dtype=wp.int32, device=device)
    H_cols = wp.array(H_cols, dtype=wp.int32, device=device)
    return H_rows, H_cols


def generate_compliance_block_indices(max_rigid_contacts: int, device: wpctx.Device):
    """
    Generate row and column indices for non-zero elements of the compliance matrix
    for rigid contact constraints.
    Args:
        max_rigid_contacts (int): Maximum number of rigid contact constraints.
        device (warp.context.Device): The device on which to create the arrays.
    Returns:
        C_rows (np.array): Row indices of non-zero elements.
        C_cols (np.array): Column indices of non-zero elements.
    """
    C_rows = np.arange(0, max_rigid_contacts, dtype=np.int32)
    C_cols = np.arange(0, max_rigid_contacts, dtype=np.int32)
    C_rows = wp.array(C_rows, dtype=wp.int32, device=device)
    C_cols = wp.array(C_cols, dtype=wp.int32, device=device)
    return C_rows, C_cols


@wp.kernel
def fill_J_n_indices_kernel(
    shape_body: wp.array(dtype=int),  # [B]
    contact_count: wp.array(dtype=wp.int32),  # [1]
    contact_shape0: wp.array(dtype=wp.int32),  # [C]
    contact_shape1: wp.array(dtype=wp.int32),  # [C]
    J_n_offset: int,  # Offset for the normal contact constraint Jacobian indices
    J_n_dense_offset: wp.vec2i,  # Offset for the normal contact constraint Jacobian
    # --- Outputs ---
    J_rows: wp.array(dtype=wp.int32),
    J_cols: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    # Handle inactive constraints (not detected by collide func)
    if tid >= contact_count[0]:
        return

    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    # Guard against self-contact
    if shape_a == shape_b:
        return

    # Get body indices and thickness
    body_a, body_b = -1, -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            idxs_offset = J_n_offset + 12 * tid
            J_rows[idxs_offset + st_i] = J_n_dense_offset.x + tid
            J_cols[idxs_offset + st_i] = J_n_dense_offset.y + 6 * body_a + st_i
    if shape_b >= 0:
        body_b = shape_body[shape_b]
        for i in range(wp.static(6)):
            st_i = wp.static(i)
            idxs_offset = J_n_offset + 12 * tid + 6
            J_rows[idxs_offset + st_i] = J_n_dense_offset.x + tid
            J_cols[idxs_offset + st_i] = J_n_dense_offset.y + 6 * body_b + st_i


def add_delta_x(
    delta_x: wp.array,
    body_qd: wp.array,
    lambda_n: wp.array,
    d_offset: int,
    n_offset: int,
):

    B = body_qd.shape[0]
    C = lambda_n.shape[0]
    add_inplace(body_qd, delta_x, 0, d_offset, B)
    add_inplace(lambda_n, delta_x, 0, n_offset, C)


class NSNEngine(Integrator):
    def __init__(
        self,
        model: Model,
        tolerance: float = 1e-3,
        max_iterations: int = 10,
        use_cuda_graph: bool = True,
    ):
        super().__init__()
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        device = model.device
        N_b = model.body_count
        N_c = model.rigid_contact_max

        # --- Residual ---
        # Momentum balance
        self.g = wp.zeros((N_b * 6,), dtype=wp.float32, device=device)
        # The vector of the constraint errors
        self.h = wp.zeros((N_c,), dtype=wp.float32, device=device)
        self.h_n_offset = 0  # Offset for the normal contact constraint errors

        # --- Jacobian ---
        # The constraint block (dense matrix)
        self.J = wps.bsr_zeros(N_c, 6 * N_b, block_type=wp.float32, device=device)
        self.J_rows = wp.zeros(
            12 * N_c, dtype=wp.int32, device=device
        )  # Needs to be computed every time
        self.J_cols = wp.zeros(
            12 * N_c, dtype=wp.int32, device=device
        )  # Needs to be computed every time
        self.J_values = wp.zeros(12 * N_c, dtype=wp.float32, device=device)
        self.J_n_offset = 0
        self.J_n_dense_offset = wp.vec2i(0, 0)

        self.J_T = wps.bsr_zeros(N_c, 6 * N_b, block_type=wp.float32, device=device)

        # Mass block (sparse matrix)
        self.H_inv = wps.bsr_zeros(
            6 * N_b, 6 * N_b, block_type=wp.float32, device=device
        )
        self.H_inv_rows, self.H_inv_cols = generate_mass_block_indices(N_b, device)
        self.H_inv_values = wp.zeros(N_b * 12, dtype=wp.float32, device=device)

        # Compliance block (sparse matrix)
        self.C = wps.bsr_zeros(N_c, N_c, block_type=wp.float32, device=device)
        self.C_rows, self.C_cols = generate_compliance_block_indices(N_c, device)
        self.C_values = wp.zeros(self.C_rows.shape[0], dtype=wp.float32, device=device)
        self.C_n_offset = 0  # Offset for the normal contact constraint compliance

        # Contact impulse vector
        self.lambda_n = wp.zeros(N_c, dtype=wp.float32, device=device)

        self.delta_body_qd = wp.zeros(6 * N_b, dtype=wp.float32, device=device)
        self.delta_lambda = wp.zeros(N_c, dtype=wp.float32, device=device)

        # A = J @ H^{-1} @ J^T + C
        # Z = J @ H^{-1}
        # C := Y @ J^T + C
        self.Z = wps.bsr_zeros(N_c, 6 * N_b, block_type=wp.float32, device=device)

        self.mm_work_arrays_1 = wps.bsr_mm_work_arrays()
        self.mm_work_arrays_2 = wps.bsr_mm_work_arrays()

        # ----------- WARM-UP -----------
        wps.bsr_set_from_triplets(
            self.H_inv, self.H_inv_rows, self.H_inv_cols, self.H_inv_values
        )
        wps.bsr_set_from_triplets(self.J, self.J_rows, self.J_cols, self.J_values)
        wps.bsr_set_from_triplets(self.C, self.C_rows, self.C_cols, self.C_values)

        wps.bsr_set_transpose(self.J_T, self.J)

        # # Warm up the sparse matrix multiplications
        # wps.bsr_mm(
        #     x=self.J,
        #     y=self.H_inv,
        #     z=self.Z,
        #     alpha=1.0,
        #     beta=0.0,
        #     work_arrays=self.mm_work_arrays_1,
        #     reuse_topology=False,
        # )  # Changes Z
        #
        # wps.bsr_mm(
        #     x=self.Z,
        #     y=self.J_T,
        #     z=self.C,
        #     alpha=1.0,
        #     beta=1.0,
        #     work_arrays=self.mm_work_arrays_2,
        #     reuse_topology=False,
        # )  # Changes C

    def _clear_values(self):
        self.g.zero_()
        self.h.zero_()
        self.H_inv_values.zero_()
        self.J_values.zero_()
        self.C_values.zero_()

    def _fill_constraint_block_indices(self, model: Model):
        N_c = model.rigid_contact_max

        wp.launch(
            kernel=fill_J_n_indices_kernel,
            dim=N_c,
            inputs=[
                model.shape_body,
                model.rigid_contact_count,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                self.J_n_offset,
                self.J_n_dense_offset,
            ],
            outputs=[self.J_rows, self.J_cols],
            device=model.device,
        )

    def _fill_matrices(
        self, model: Model, state_in: State, state_out: State, dt: float
    ):
        self._clear_values()
        self._fill_constraint_block_indices(
            model
        )  # TODO: Doesn't need to be called every time

        N_b = model.body_count
        N_c = model.rigid_contact_max

        # Compute the dynamics contact constraint
        wp.launch(
            kernel=unconstrained_dynamics_kernel,
            dim=N_b,
            inputs=[
                state_out.body_qd,
                state_in.body_qd,
                state_out.body_f,
                model.body_mass,
                model.body_inv_mass,
                model.body_inertia,
                model.body_inv_inertia,
                dt,
                model.gravity,
            ],
            outputs=[self.g, self.H_inv_values],
            device=model.device,
        )

        # Compute the contact constraint
        wp.launch(
            kernel=contact_constraint_kernel,
            dim=N_c,
            inputs=[
                state_out.body_q,
                state_out.body_qd,
                state_in.body_qd,
                model.body_com,
                model.shape_body,
                model.shape_geo,
                model.shape_materials,
                model.rigid_contact_count,
                model.rigid_contact_point0,
                model.rigid_contact_point1,
                model.rigid_contact_normal,
                model.rigid_contact_shape0,
                model.rigid_contact_shape1,
                self.lambda_n,
                dt,
                CONTACT_CONSTRAINT_STABILIZATION,
                CONTACT_FB_ALPHA,
                CONTACT_FB_BETA,
                self.h_n_offset,
                self.J_n_offset,
                self.C_n_offset,
            ],
            outputs=[self.g, self.h, self.J_values, self.C_values],
        )

        # Fill the sparse matrices
        wps.bsr_set_from_triplets(
            self.H_inv,
            self.H_inv_rows,
            self.H_inv_cols,
            self.H_inv_values,
            prune_numerical_zeros=True,
        )
        wps.bsr_set_from_triplets(
            self.J,
            self.J_rows,
            self.J_cols,
            self.J_values,
            prune_numerical_zeros=True,
        )
        wps.bsr_set_from_triplets(
            self.C,
            self.C_rows,
            self.C_cols,
            self.C_values,
            prune_numerical_zeros=True,
        )

        wps.bsr_set_transpose(self.J_T, self.J)

    def simulate(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        control: Control | None = None,
    ):
        N_b = model.body_count
        N_c = model.rigid_contact_max

        if N_b == 0:
            raise ValueError("State must contain at least one body.")

        if state_in.particle_count > 0:
            raise ValueError("NSNEngine does not support particles.")

        if control is None:
            control = model.control(clone_variables=False)

        # Get the initial guess for the output state. This will be used as the starting point for the iterative solver.
        self.integrate_bodies(model, state_in, state_out, dt)

        for _ in range(self.max_iterations):
            with wp.ScopedTimer("Fill Matrices", active=True):
                self._fill_matrices(model, state_in, state_out, dt)

            with wp.ScopedTimer("MM", active=True):

                # A = J @ H^{-1} @ J^T + C
                wps.bsr_mm(
                    x=self.J,
                    y=self.H_inv,
                    z=self.Z,
                    alpha=1.0,
                    beta=0.0,
                    # work_arrays=self.mm_work_arrays_1,
                    # reuse_topology=True,
                )  # Changes Z

                wps.bsr_mm(
                    x=self.Z,
                    y=self.J_T,
                    z=self.C,
                    alpha=1.0,
                    beta=1.0,
                    # work_arrays=self.mm_work_arrays_2,
                    # reuse_topology=True,
                )  # Changes C
            with wp.ScopedTimer("Solve", active=True):
                # b = J @ H^{-1} @ g - h
                wps.bsr_mv(
                    A=self.Z, x=self.g, y=self.h, alpha=1.0, beta=-1.0
                )  # Changes h

                # Solve the linear system A * delta_lambda = b
                M = wpol.preconditioner(self.Z, ptype="diag")
                _ = cr_solver_graph_compatible(
                    A=self.C,
                    b=self.h,
                    x=self.delta_lambda,
                    iters=10,
                    M=M,
                )

                # g := J^T @ Δλ - g
                wps.bsr_mv(
                    A=self.J_T, x=self.delta_lambda, y=self.g, alpha=1.0, beta=-1.0
                )  # Changes g

                # Δu := H^{-1} @ g
                wps.bsr_mv(
                    A=self.H_inv, x=self.g, y=self.delta_body_qd, alpha=1.0, beta=0.0
                )  # Changes delta_body_qd

                # Add the changes to the state
                add_inplace(state_out.body_qd, self.delta_body_qd, 0, 0, N_b)
                add_inplace(self.lambda_n, self.delta_lambda, 0, 0, N_c)
