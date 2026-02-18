import warp as wp
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions


class TrajectoryBuffer:
    def __init__(self, dims: EngineDimensions, num_steps: int, device):
        self.num_steps = num_steps
        self.dims = dims
        self.device = device

        def _alloc(shape, dtype, requires_grad=False):
            return wp.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)

        # =========================================================================
        # 1. Body & Joint State (Requires N+1 slots for Time 0..N)
        # =========================================================================
        # State exists at boundaries (Start -> End)
        per_body_shape_larger = (num_steps + 1, dims.num_worlds, dims.body_count)
        per_body_shape = (num_steps, dims.num_worlds, dims.body_count)

        self.body_pose = _alloc(per_body_shape_larger, wp.transform, True)
        self.body_vel = _alloc(per_body_shape_larger, wp.spatial_vector, True)
        # Forces apply DURING the step -> Size N
        self.ext_force = _alloc(per_body_shape, wp.spatial_vector, True)

        per_joint_dof_shape = (num_steps, dims.num_worlds, dims.joint_dof_count)

        self.joint_target_pos = _alloc(per_joint_dof_shape, wp.float32, True)
        self.joint_target_vel = _alloc(per_joint_dof_shape, wp.float32, True)

        # =========================================================================
        # 2. Constraint Multipliers (Interval Data -> Size N)
        # =========================================================================
        per_constraint_shape = (num_steps, dims.num_worlds, dims.N_c)
        per_constraint_idx_shape = (num_steps, dims.num_worlds, dims.N_c, 2)

        self.constr_force = _alloc(per_constraint_shape, wp.float32)
        self.constr_force_prev_iter = _alloc(per_constraint_shape, wp.float32)
        self.constr_active_mask = _alloc(per_constraint_shape, wp.float32)
        self.constr_body_idx = _alloc(per_constraint_idx_shape, wp.int32)

        # =========================================================================
        # 3. Contact Manifold (Interval Data -> Size N)
        # =========================================================================
        per_contact_shape = (num_steps, dims.num_worlds, dims.contact_count)

        self.contact_body_a = _alloc(per_contact_shape, wp.int32)
        self.contact_body_b = _alloc(per_contact_shape, wp.int32)
        self.contact_point_a = _alloc(per_contact_shape, wp.vec3)
        self.contact_point_b = _alloc(per_contact_shape, wp.vec3)
        self.contact_normal = _alloc(per_contact_shape, wp.spatial_vector)
        self.contact_dist = _alloc(per_contact_shape, wp.float32)
        self.contact_friction = _alloc(per_contact_shape, wp.float32)
        self.contact_restitution = _alloc(per_contact_shape, wp.float32)
        self.contact_thickness_a = _alloc(per_contact_shape, wp.float32)
        self.contact_thickness_b = _alloc(per_contact_shape, wp.float32)
        self.contact_basis_t1_a = _alloc(per_contact_shape, wp.spatial_vector)
        self.contact_basis_t2_a = _alloc(per_contact_shape, wp.spatial_vector)
        self.contact_basis_n_b = _alloc(per_contact_shape, wp.spatial_vector)
        self.contact_basis_t1_b = _alloc(per_contact_shape, wp.spatial_vector)
        self.contact_basis_t2_b = _alloc(per_contact_shape, wp.spatial_vector)

    def zero_grad(self):
        if self.body_pose.requires_grad:
            self.body_pose.grad.zero_()
        if self.body_vel.requires_grad:
            self.body_vel.grad.zero_()
        if self.ext_force.requires_grad:
            self.ext_force.grad.zero_()
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
            wp.copy(self.body_pose[0], data.body_pose_prev)
            wp.copy(self.body_vel[0], data.body_vel_prev)

        # 2. Body State (Result of step t goes to t+1)
        wp.copy(self.body_pose[step_idx + 1], data.body_pose)
        wp.copy(self.body_vel[step_idx + 1], data.body_vel)

        # 3. Interval Data (Forces applied during step t go to t)
        wp.copy(self.ext_force[step_idx], data.ext_force)

        # --- Inputs ---
        wp.copy(self.joint_target_pos[step_idx], data.joint_target_pos)
        wp.copy(self.joint_target_vel[step_idx], data.joint_target_vel)

        # --- Lambdas ---
        wp.copy(self.constr_force[step_idx], data._constr_force)
        wp.copy(self.constr_force_prev_iter[step_idx], data._constr_force_prev_iter)
        wp.copy(self.constr_active_mask[step_idx], data._constr_active_mask)
        wp.copy(self.constr_body_idx[step_idx], data._constr_body_idx)

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
        wp.copy(data.body_pose, self.body_pose[step_idx + 1])  # Load Result
        wp.copy(data.body_pose_prev, self.body_pose[step_idx])  # Load Start

        wp.copy(data.body_vel, self.body_vel[step_idx + 1])
        wp.copy(data.body_vel_prev, self.body_vel[step_idx])

        wp.copy(data.body_pose_grad, self.body_pose.grad[step_idx + 1])
        wp.copy(data.body_vel_grad, self.body_vel.grad[step_idx + 1])

        wp.copy(data.ext_force, self.ext_force[step_idx])

        # --- Inputs ---
        wp.copy(data.joint_target_pos, self.joint_target_pos[step_idx])
        wp.copy(data.joint_target_vel, self.joint_target_vel[step_idx])

        # --- Lambdas ---
        wp.copy(data._constr_force, self.constr_force[step_idx])
        wp.copy(data._constr_force_prev_iter, self.constr_force_prev_iter[step_idx])
        wp.copy(data._constr_active_mask, self.constr_active_mask[step_idx])
        wp.copy(data._constr_body_idx, self.constr_body_idx[step_idx])

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
        wp.copy(self.body_pose.grad[step_idx], data.body_pose_prev.grad)
        wp.copy(self.body_vel.grad[step_idx], data.body_vel_prev.grad)

        # Forces and inputs are interval-based (index T)
        wp.copy(self.ext_force.grad[step_idx], data.ext_force.grad)

        # --- Inputs ---
        wp.copy(self.joint_target_pos.grad[step_idx], data.joint_target_pos.grad)
        wp.copy(self.joint_target_vel.grad[step_idx], data.joint_target_vel.grad)
