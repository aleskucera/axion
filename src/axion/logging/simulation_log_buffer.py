import h5py
import warp as wp

from .data_views import ConstraintView
from .data_views import SystemView
from .engine_config import EngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions


@wp.kernel
def increment_nr_iter(nr_iter: wp.array(dtype=wp.int32)):
    pass


@wp.kernel
def increment_step_idx():
    pass


class SimulationLogBuffer:
    def __init__(
        self,
        num_steps: int,
        data: EngineData,
        config: EngineConfig,
        dims: EngineDimensions,
        device: wp.Device,
    ):
        self.device = device
        self.dims = dims
        self.num_steps = num_steps
        self.config = config

        self.dt = data.dt if data.dt is not None else 0.0

        # --- Helper for concise allocation ---
        def _alloc_buffer(buffer_size, array):
            return wp.zeros((buffer_size,) + array.shape, dtype=array.dtype, device=device)

        # =========================================================================
        # Body State Arrays
        # =========================================================================
        self.ext_force = _alloc_buffer(num_steps, data.ext_force)
        self.body_pose = _alloc_buffer(num_steps, data.body_pose)
        self.body_vel = _alloc_buffer(num_steps, data.body_vel)
        self.joint_target_pos = _alloc_buffer(num_steps, data.joint_target_pos)
        self.joint_target_vel = _alloc_buffer(num_steps, data.joint_target_vel)

        # =========================================================================
        # Constraint Arrays & Views
        # =========================================================================
        self._constr_force = _alloc_buffer(num_steps, data._constr_force)
        self._constr_body_idx = _alloc_buffer(num_steps, data._constr_body_idx)
        self._constr_active_mask = _alloc_buffer(num_steps, data._constr_active_mask)

        # Views for constraint data
        self.constr_force = ConstraintView(self._constr_force, dims)
        self.constr_force_prev_iter = ConstraintView(self._constr_force_prev_iter, dims)
        self.constr_body_idx = ConstraintView(self._constr_body_idx, dims, axis=-2)
        self.constr_active_mask = ConstraintView(self._constr_active_mask, dims)

        # =========================================================================
        # Contact Data
        # =========================================================================
        self.contact_body_a = _alloc_buffer(num_steps, data.contact_body_a)
        self.contact_body_b = _alloc_buffer(num_steps, data.contact_body_b)
        self.contact_point_a = _alloc_buffer(num_steps, data.contact_point_a)
        self.contact_point_b = _alloc_buffer(num_steps, data.contact_point_b)
        self.contact_thickness_a = _alloc_buffer(num_steps, data.contact_thickness_a)
        self.contact_thickness_b = _alloc_buffer(num_steps, data.contact_thickness_b)
        self.contact_dist = _alloc_buffer(num_steps, data.contact_dist)
        self.contact_friction_coeff = _alloc_buffer(num_steps, data.contact_friction_coeff)
        self.contact_restitution_coeff = _alloc_buffer(num_steps, data.contact_restitution_coeff)

        self.contact_basis_n_a = _alloc_buffer(num_steps, data.contact_basis_n_a)
        self.contact_basis_t1_a = _alloc_buffer(num_steps, data.contact_basis_t1_a)
        self.contact_basis_t2_a = _alloc_buffer(num_steps, data.contact_basis_t2_a)
        self.contact_basis_n_b = _alloc_buffer(num_steps, data.contact_basis_n_b)
        self.contact_basis_t1_b = _alloc_buffer(num_steps, data.contact_basis_t1_b)
        self.contact_basis_t2_b = _alloc_buffer(num_steps, data.contact_basis_t2_b)

        # =========================================================================
        # Linear System Arrays & Views
        # =========================================================================
        self.world_M = _alloc_buffer(num_steps, data.world_M)
        self.world_M_inv = _alloc_buffer(num_steps, data.world_M_inv)

        self._J_values = _alloc_buffer(num_steps, data._J_values)
        self._C_values = _alloc_buffer(num_steps, data._C_values)

        self.J_values = ConstraintView(self._J_values, dims, axis=-2)
        self.C_values = ConstraintView(self._C_values, dims)

        # System Views
        self._res = _alloc_buffer(num_steps, data._res)
        self.res = SystemView(self._res, dims, self._res_spatial)

        # =========================================================================
        # Newton-Raphson (NR) History Arrays
        # =========================================================================
        nr_buffer_size = num_steps * (config.max_newton_iters + 1)
        self.nr_history_body_pose = _alloc_buffer(nr_buffer_size, self.body_pose)
        self.nr_history_body_vel = _alloc_buffer(nr_buffer_size, self.body_vel)

        self._nr_history_constr_force = _alloc_buffer(nr_buffer_size, self._constr_force)
        self.nr_history_constr_force = ConstraintView(self._nr_history_constr_force, dims)

        self._nr_history_res = _alloc_buffer(nr_buffer_size, self._res)
        self.nr_history_res = SystemView(self._nr_history_res, dims)

        # =========================================================================
        # Preconditioned Conjugate Residual (PCR) History Arrays
        # =========================================================================
        self.pcr_iter_count = wp.zeros((nr_buffer_size,), wp.int32)
        self.pcr_final_r_norm_sq = wp.zeros((nr_buffer_size, dims.num_worlds), wp.float32)
        self.pcr_r_norm_sq_history = wp.zeros(
            (nr_buffer_size, config.max_linear_iters + 1, dims.num_worlds), wp.float32
        )

        # =========================================================================
        # Linesearch Arrays & Views
        # =========================================================================
        if config.enable_linesearch:
            self.linesearch_step_size = _alloc_buffer(nr_buffer_size, data.linesearch_step_size)

            self.linesearch_res_norm_sq = _alloc_buffer(nr_buffer_size, data.linesearch_res_norm_sq)
            self.linesearch_minimal_index = _alloc_buffer(
                nr_buffer_size, data.linesearch_minimal_index
            )

    def log_step_end(self, step_idx: int, data: EngineData):
        # --- Body State ---
        wp.copy(self.ext_force[step_idx], data.ext_force)
        wp.copy(self.body_pose[step_idx], data.body_pose)
        wp.copy(self.body_vel[step_idx], data.body_vel)
        wp.copy(self.joint_target_pos[step_idx], data.joint_target_pos)
        wp.copy(self.joint_target_vel[step_idx], data.joint_target_vel)

        # --- Constraints ---
        wp.copy(self._constr_force[step_idx], data._constr_force)
        wp.copy(self._constr_body_idx[step_idx], data._constr_body_idx)
        wp.copy(self._constr_active_mask[step_idx], data._constr_active_mask)

        # --- Contact Data ---
        wp.copy(self.contact_body_a[step_idx], data.contact_body_a)
        wp.copy(self.contact_body_b[step_idx], data.contact_body_b)
        wp.copy(self.contact_point_a[step_idx], data.contact_point_a)
        wp.copy(self.contact_point_b[step_idx], data.contact_point_b)
        wp.copy(self.contact_thickness_a[step_idx], data.contact_thickness_a)
        wp.copy(self.contact_thickness_b[step_idx], data.contact_thickness_b)
        wp.copy(self.contact_dist[step_idx], data.contact_dist)
        wp.copy(self.contact_friction_coeff[step_idx], data.contact_friction_coeff)
        wp.copy(self.contact_restitution_coeff[step_idx], data.contact_restitution_coeff)

        wp.copy(self.contact_basis_n_a[step_idx], data.contact_basis_n_a)
        wp.copy(self.contact_basis_t1_a[step_idx], data.contact_basis_t1_a)
        wp.copy(self.contact_basis_t2_a[step_idx], data.contact_basis_t2_a)
        wp.copy(self.contact_basis_n_b[step_idx], data.contact_basis_n_b)
        wp.copy(self.contact_basis_t1_b[step_idx], data.contact_basis_t1_b)
        wp.copy(self.contact_basis_t2_b[step_idx], data.contact_basis_t2_b)

        # --- Linear System Arrays ---
        wp.copy(self.world_M[step_idx], data.world_M)
        wp.copy(self.world_M_inv[step_idx], data.world_M_inv)
        wp.copy(self._J_values[step_idx], data._J_values)
        wp.copy(self._C_values[step_idx], data._C_values)
        wp.copy(self._res[step_idx], data._res)

    def log_nr_iteration(
        self,
        step_idx: int,
        nr_iter: int,
        data: EngineData,
        pcr_iter_count: wp.array,
        pcr_final_r_norm_sq: wp.array,
        pcr_r_norm_sq_history: wp.array,
    ):
        # --- Body State ---
        wp.copy(self.body_pose[step_idx * nr_iter], data.body_pose)
        wp.copy(self.body_vel[step_idx * nr_iter], data.body_vel)
        wp.copy(self._constr_force[step_idx * nr_iter], data._constr_force)
        wp.copy(self._res[step_idx * nr_iter], data._res)

        # --- Linesearch ---
        if self.config.enable_linesearch:
            wp.copy(self.linesearch_step_size[step_idx * nr_iter], data.linesearch_step_size)
            wp.copy(self.linesearch_res_norm_sq[step_idx * nr_iter], data.linesearch_res_norm_sq)
            wp.copy(
                self.linesearch_minimal_index[step_idx * nr_iter], data.linesearch_minimal_index
            )

        # Also create some sort of the pcr mask
        wp.copy(self.pcr_iter_count[step_idx * nr_iter], pcr_iter_count)
        wp.copy(self.pcr_final_r_norm_sq[step_idx * nr_iter], pcr_final_r_norm_sq)
        wp.copy(self.pcr_r_norm_sq_history[step_idx * nr_iter], pcr_r_norm_sq_history)

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
            f.attrs["dt"] = self.dt

            # Automate saving of all wp.arrays in this class
            for attr_name, attr_value in vars(self).items():
                if isinstance(attr_value, wp.array):
                    # 1. Sync to CPU (numpy)
                    # We use .numpy() to pull data from GPU
                    data_np = attr_value.numpy()

                    # 2. Create Dataset
                    # Compressing zeros is very efficient
                    f.create_dataset(attr_name, data=data_np, compression="gzip")

        print("Save complete.")
