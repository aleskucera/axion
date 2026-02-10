import warp as wp
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions


class TrajectoryBuffer:
    def __init__(self, dims: EngineDimensions, num_steps: int, device):
        self.num_steps = num_steps
        self.dims = dims
        self.device = device

        # =========================================================================
        # 1. Body & Joint State (Requires N+1 slots for Time 0..N)
        # =========================================================================
        # State exists at boundaries (Start -> End)
        per_body_shape_larger = (num_steps + 1, dims.num_worlds, dims.body_count)
        per_body_shape = (num_steps, dims.num_worlds, dims.body_count)

        with wp.ScopedDevice(device):
            self.body_q = wp.zeros(per_body_shape_larger, dtype=wp.transform, requires_grad=True)
            self.body_u = wp.zeros(
                per_body_shape_larger, dtype=wp.spatial_vector, requires_grad=True
            )

            # Forces apply DURING the step -> Size N
            self.body_f = wp.zeros(per_body_shape, dtype=wp.spatial_vector, requires_grad=True)

        per_joint_dof_shape = (num_steps, dims.num_worlds, dims.joint_dof_count)
        with wp.ScopedDevice(device):
            self.joint_target_pos = wp.zeros(
                per_joint_dof_shape, dtype=wp.float32, requires_grad=True
            )
            self.joint_target_vel = wp.zeros(
                per_joint_dof_shape, dtype=wp.float32, requires_grad=True
            )

        # =========================================================================
        # 2. Constraint Multipliers (Interval Data -> Size N)
        # =========================================================================
        per_constraint_shape = (num_steps, dims.num_worlds, dims.N_c)
        per_constraint_idx_shape = (num_steps, dims.num_worlds, dims.N_c, 2)

        with wp.ScopedDevice(device):
            self.body_lambda = wp.zeros(per_constraint_shape, dtype=wp.float32)
            self.body_lambda_prev = wp.zeros(per_constraint_shape, dtype=wp.float32)
            self.constraint_active_mask = wp.zeros(per_constraint_shape, dtype=wp.float32)
            self.constraint_body_idx = wp.zeros(per_constraint_idx_shape, dtype=wp.int32)

        # =========================================================================
        # 3. Contact Manifold (Interval Data -> Size N)
        # =========================================================================
        per_contact_shape = (num_steps, dims.num_worlds, dims.contact_count)
        with wp.ScopedDevice(device):
            self.contact_body_a = wp.zeros(per_contact_shape, dtype=wp.int32)
            self.contact_body_b = wp.zeros(per_contact_shape, dtype=wp.int32)
            self.contact_point_a = wp.zeros(per_contact_shape, dtype=wp.vec3)
            self.contact_point_b = wp.zeros(per_contact_shape, dtype=wp.vec3)
            self.contact_normal = wp.zeros(per_contact_shape, dtype=wp.spatial_vector)
            self.contact_dist = wp.zeros(per_contact_shape, dtype=wp.float32)
            self.contact_friction = wp.zeros(per_contact_shape, dtype=wp.float32)
            self.contact_restitution = wp.zeros(per_contact_shape, dtype=wp.float32)
            self.contact_thickness_a = wp.zeros(per_contact_shape, dtype=wp.float32)
            self.contact_thickness_b = wp.zeros(per_contact_shape, dtype=wp.float32)
            self.contact_basis_t1_a = wp.zeros(per_contact_shape, dtype=wp.spatial_vector)
            self.contact_basis_t2_a = wp.zeros(per_contact_shape, dtype=wp.spatial_vector)
            self.contact_basis_n_b = wp.zeros(per_contact_shape, dtype=wp.spatial_vector)
            self.contact_basis_t1_b = wp.zeros(per_contact_shape, dtype=wp.spatial_vector)
            self.contact_basis_t2_b = wp.zeros(per_contact_shape, dtype=wp.spatial_vector)

    def zero_grad(self):
        if self.body_q.requires_grad:
            self.body_q.grad.zero_()
        if self.body_u.requires_grad:
            self.body_u.grad.zero_()
        if self.body_f.requires_grad:
            self.body_f.grad.zero_()
        if self.joint_target_pos.requires_grad:
            self.joint_target_pos.grad.zero_()
        if self.joint_target_vel.requires_grad:
            self.joint_target_vel.grad.zero_()

    def save_step(self, step_idx: int, data: EngineData):
        """
        Saves the current state from EngineData into the buffer.
        State Result -> Index [step_idx + 1]
        Interval Data -> Index [step_idx]
        """
        assert step_idx >= 0, "Argument 'step_idx' has to be larger or equal to zero."

        # 1. Handle Initial Conditions (Only on first step)
        if step_idx == 0:
            wp.copy(self.body_q[0], data.body_q_prev)
            wp.copy(self.body_u[0], data.body_u_prev)

        # 2. Body State (Result of step t goes to t+1)
        wp.copy(self.body_q[step_idx + 1], data.body_q)
        wp.copy(self.body_u[step_idx + 1], data.body_u)

        # 3. Interval Data (Forces applied during step t go to t)
        wp.copy(self.body_f[step_idx], data.body_f)

        # --- Inputs ---
        wp.copy(self.joint_target_pos[step_idx], data.joint_target_pos)
        wp.copy(self.joint_target_vel[step_idx], data.joint_target_vel)

        # --- Lambdas ---
        wp.copy(self.body_lambda[step_idx], data._body_lambda)
        wp.copy(self.body_lambda_prev[step_idx], data._body_lambda_prev)
        wp.copy(self.constraint_active_mask[step_idx], data._constraint_active_mask)
        wp.copy(self.constraint_body_idx[step_idx], data._constraint_body_idx)

        # --- Contact Manifold ---
        wp.copy(self.contact_body_a[step_idx], data.contact_body_a)
        wp.copy(self.contact_body_b[step_idx], data.contact_body_b)
        wp.copy(self.contact_point_a[step_idx], data.contact_point_a)
        wp.copy(self.contact_point_b[step_idx], data.contact_point_b)
        wp.copy(self.contact_normal[step_idx], data.contact_basis_n_a)  # Renamed field
        wp.copy(self.contact_dist[step_idx], data.contact_dist)
        wp.copy(self.contact_friction[step_idx], data.contact_friction_coeff)
        wp.copy(self.contact_restitution[step_idx], data.contact_restitution_coeff)
        wp.copy(self.contact_thickness_a[step_idx], data.contact_thickness_a)
        wp.copy(self.contact_thickness_b[step_idx], data.contact_thickness_b)
        wp.copy(self.contact_basis_t1_a[step_idx], data.contact_basis_t1_a)
        wp.copy(self.contact_basis_t2_a[step_idx], data.contact_basis_t2_a)
        wp.copy(self.contact_basis_n_b[step_idx], data.contact_basis_n_b)
        wp.copy(self.contact_basis_t1_b[step_idx], data.contact_basis_t1_b)
        wp.copy(self.contact_basis_t2_b[step_idx], data.contact_basis_t2_b)

    def load_step(self, step_idx: int, data: EngineData):
        """
        Restores the state from the buffer into EngineData.
        Start State <- Index [step_idx]
        Result State <- Index [step_idx + 1]
        """
        # --- Body State ---
        wp.copy(data.body_q, self.body_q[step_idx + 1])  # Load Result
        wp.copy(data.body_q_prev, self.body_q[step_idx])  # Load Start

        wp.copy(data.body_u, self.body_u[step_idx + 1])
        wp.copy(data.body_u_prev, self.body_u[step_idx])

        wp.copy(data.body_q_grad, self.body_q.grad[step_idx + 1])
        wp.copy(data.body_u_grad, self.body_u.grad[step_idx + 1])

        wp.copy(data.body_f, self.body_f[step_idx])

        # --- Inputs ---
        wp.copy(data.joint_target_pos, self.joint_target_pos[step_idx])
        wp.copy(data.joint_target_vel, self.joint_target_vel[step_idx])

        # --- Lambdas ---
        wp.copy(data._body_lambda, self.body_lambda[step_idx])
        wp.copy(data._body_lambda_prev, self.body_lambda_prev[step_idx])
        wp.copy(data._constraint_active_mask, self.constraint_active_mask[step_idx])
        wp.copy(data._constraint_body_idx, self.constraint_body_idx[step_idx])

        # --- Contact Manifold ---
        wp.copy(data.contact_body_a, self.contact_body_a[step_idx])
        wp.copy(data.contact_body_b, self.contact_body_b[step_idx])
        wp.copy(data.contact_point_a, self.contact_point_a[step_idx])
        wp.copy(data.contact_point_b, self.contact_point_b[step_idx])
        wp.copy(data.contact_basis_n_a, self.contact_normal[step_idx])
        wp.copy(data.contact_dist, self.contact_dist[step_idx])
        wp.copy(data.contact_friction_coeff, self.contact_friction[step_idx])
        wp.copy(data.contact_restitution_coeff, self.contact_restitution[step_idx])
        wp.copy(data.contact_thickness_a, self.contact_thickness_a[step_idx])
        wp.copy(data.contact_thickness_b, self.contact_thickness_b[step_idx])
        wp.copy(data.contact_basis_t1_a, self.contact_basis_t1_a[step_idx])
        wp.copy(data.contact_basis_t2_a, self.contact_basis_t2_a[step_idx])
        wp.copy(data.contact_basis_n_b, self.contact_basis_n_b[step_idx])
        wp.copy(data.contact_basis_t1_b, self.contact_basis_t1_b[step_idx])
        wp.copy(data.contact_basis_t2_b, self.contact_basis_t2_b[step_idx])

    def save_gradients(self, step_idx: int, data: EngineData):
        """
        Saves gradients computed during a backward pass from EngineData into the buffer.
        """
        # --- Body State ---
        # Note: We save the gradients w.r.t the INITIAL state of this step (q_prev)
        # into the buffer at step_idx (which corresponds to q at time T)
        wp.copy(self.body_q.grad[step_idx], data.body_q_prev.grad)
        wp.copy(self.body_u.grad[step_idx], data.body_u_prev.grad)

        # Forces and inputs are interval-based (index T)
        wp.copy(self.body_f.grad[step_idx], data.body_f.grad)

        # --- Inputs ---
        wp.copy(self.joint_target_pos.grad[step_idx], data.joint_target_pos.grad)
        wp.copy(self.joint_target_vel.grad[step_idx], data.joint_target_vel.grad)
