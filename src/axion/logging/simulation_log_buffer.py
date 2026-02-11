import h5py
import numpy as np
import warp as wp
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions


class SimulationLogBuffer:
    def __init__(self, dims: EngineDimensions, num_steps: int, device: wp.Device):
        self.dims = dims
        self.num_steps = num_steps
        self.device = device

        # --- Helper for concise allocation ---
        def _alloc(shape, dtype, requires_grad):
            # We add (num_steps, ...) to the front
            full_shape = (num_steps,) + shape
            return wp.zeros(full_shape, dtype=dtype, device=device, requires_grad=False)

        self.h = _alloc((dims.num_worlds, dims.N_u + dims.num_constraints), wp.float32)

        # 2. Separate Spatial Buffer Allocation
        self.h_spatial = _alloc((dims.num_worlds, dims.body_count), wp.spatial_vector)

        self.J_values = _alloc((dims.num_worlds, dims.num_constraints, 2), wp.spatial_vector)
        self.C_values = _alloc((dims.num_worlds, dims.num_constraints))

        self.body_f = _alloc((dims.num_worlds, dims.body_count), wp.spatial_vector)
        self.body_q = _alloc((dims.num_worlds, dims.body_count), wp.transform)
        self.body_q_prev = _alloc((dims.num_worlds, dims.body_count), wp.transform)
        self.body_u = _alloc((dims.num_worlds, dims.body_count), wp.spatial_vector)
        self.body_u_prev = _alloc((dims.num_worlds, dims.body_count), wp.spatial_vector)

        self.body_lambda = _alloc((dims.num_worlds, dims.num_constraints))
        self.body_lambda_prev = _alloc((dims.num_worlds, dims.num_constraints))

        self.s_n = _alloc((dims.num_worlds, dims.contact_count))
        self.s_n_prev = _alloc((dims.num_worlds, dims.contact_count))

        self.constraint_body_idx = _alloc((dims.num_worlds, dims.num_constraints, 2), wp.int32)
        self.constraint_active_mask = _alloc((dims.num_worlds, dims.num_constraints), wp.float32)

        self.JT_delta_lambda = _alloc((dims.num_worlds, dims.body_count), wp.spatial_vector)
        self.system_diag = _alloc((dims.num_worlds, dims.num_constraints), wp.float32)
        self.dbody_u = _alloc((dims.num_worlds, dims.body_count), wp.spatial_vector)
        self.dbody_lambda = _alloc((dims.num_worlds, dims.num_constraints))

        self.b = _alloc((dims.num_worlds, dims.num_constraints))

        self.contact_body_a = _alloc((dims.num_worlds, dims.contact_count), wp.int32)
        self.contact_body_b = _alloc((dims.num_worlds, dims.contact_count), wp.int32)
        self.contact_point_a = _alloc((dims.num_worlds, dims.contact_count), wp.vec3)
        self.contact_point_b = _alloc((dims.num_worlds, dims.contact_count), wp.vec3)
        self.contact_thickness_a = _alloc((dims.num_worlds, dims.contact_count), wp.float32)
        self.contact_thickness_b = _alloc((dims.num_worlds, dims.contact_count), wp.float32)
        self.contact_dist = _alloc((dims.num_worlds, dims.contact_count), wp.float32)
        self.contact_friction_coeff = _alloc((dims.num_worlds, dims.contact_count), wp.float32)
        self.contact_restitution_coeff = _alloc((dims.num_worlds, dims.contact_count), wp.float32)
        self.contact_basis_n_a = _alloc((dims.num_worlds, dims.contact_count), wp.spatial_vector)
        self.contact_basis_t1_a = _alloc((dims.num_worlds, dims.contact_count), wp.spatial_vector)
        self.contact_basis_t2_a = _alloc((dims.num_worlds, dims.contact_count), wp.spatial_vector)
        self.contact_basis_n_b = _alloc((dims.num_worlds, dims.contact_count), wp.spatial_vector)
        self.contact_basis_t1_b = _alloc((dims.num_worlds, dims.contact_count), wp.spatial_vector)
        self.contact_basis_t2_b = _alloc((dims.num_worlds, dims.contact_count), wp.spatial_vector)

        self.joint_target_pos = _alloc((dims.num_worlds, dims.joint_dof_count))
        self.joint_target_vel = _alloc((dims.num_worlds, dims.joint_dof_count))

        self.w = _alloc((dims.num_worlds, dims.N_u + dims.num_constraints))
        self.w_spatial = _alloc((dims.num_worlds, dims.body_count), wp.spatial_vector)
        self.adjoint_rhs = _alloc((dims.num_worlds, dims.num_constraints))

    def capture(self, step_idx: int, data: EngineData):
        """
        Records the copy operations into the current CUDA graph.
        Call this inside your simulation loop.
        """
        # Dynamics
        wp.copy(self.body_q[step_idx], data.body_q)
        wp.copy(self.body_u[step_idx], data.body_u)
        wp.copy(self.body_f[step_idx], data.body_f)

        # Constraints
        wp.copy(self.body_lambda[step_idx], data._body_lambda)
        if self.dims.N_n > 0:
            wp.copy(self.s_n[step_idx], data.s_n)

            # Contacts
            # Note: Contacts in EngineData are often "max_contacts" sized.
            # We copy the whole buffer even if not all are active,
            # because dynamic slicing is hard in Graphs.
            wp.copy(self.contact_dist[step_idx], data.contact_dist)
            wp.copy(self.contact_body_a[step_idx], data.contact_body_a)
            wp.copy(self.contact_body_b[step_idx], data.contact_body_b)
            wp.copy(self.contact_point_a[step_idx], data.contact_point_a)

    def save_to_hdf5(self, filepath: str):
        """
        Syncs data to CPU and saves to HDF5.
        Call this AFTER the simulation loop finishes.
        """
        print(f"Saving simulation log to {filepath}...")

        with h5py.File(filepath, "w") as f:
            # Metadata
            f.attrs["num_steps"] = self.num_steps
            f.attrs["num_worlds"] = self.dims.num_worlds

            # Automate saving of all wp.arrays in this class
            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, wp.array):
                    # 1. Sync to CPU (numpy)
                    data_np = attr_value.numpy()

                    # 2. Create Dataset (Structure of Arrays)
                    # Compressing zeros is very efficient
                    f.create_dataset(attr_name, data=data_np, compression="gzip")

        print("Save complete.")
