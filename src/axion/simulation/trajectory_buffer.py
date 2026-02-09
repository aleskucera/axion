import warp as wp
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions


class TrajectoryBuffer:
    def __init__(self, dims: EngineDimensions, num_steps: int, device):
        self.num_steps = num_steps
        self.dims = dims
        self.device = device

        # =========================================================================
        # 1. Body & Joint State (Fixed Size)
        # =========================================================================
        per_body_shape = (num_steps, dims.num_worlds, dims.body_count)
        per_body_shape_larger = (num_steps + 1, dims.num_worlds, dims.body_count)
        with wp.ScopedDevice(device):
            self.body_q = wp.zeros(per_body_shape_larger, dtype=wp.transform, requires_grad=True)
            # self.body_q_prev = wp.zeros(per_body_shape, dtype=wp.transform)
            self.body_u = wp.zeros(
                per_body_shape_larger, dtype=wp.spatial_vector, requires_grad=True
            )
            # self.body_u_prev = wp.zeros(per_body_shape, dtype=wp.spatial_vector)
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
        # 2. Constraint Multipliers (Lambdas)
        # =========================================================================
        # We store the raw underlying lambda array which contains [Joints | Control | Contact | Friction]
        # This is more efficient than copying views separately.
        per_constraint_shape = (num_steps, dims.num_worlds, dims.N_c)
        per_constraint_shape_2 = (num_steps, dims.num_worlds, dims.N_c, 2)
        per_constraint_shape_larger = (num_steps + 1, dims.num_worlds, dims.N_c)
        with wp.ScopedDevice(device):
            self.body_lambda = wp.zeros(per_constraint_shape_larger, dtype=wp.float32)
            self.body_lambda_prev = wp.zeros(per_constraint_shape_larger, dtype=wp.float32)
            self.constraint_active_mask = wp.zeros(per_constraint_shape, dtype=wp.float32)
            self.constraint_body_idx = wp.zeros(per_constraint_shape_2, dtype=wp.int32)

        # =========================================================================
        # 3. Contact Manifold (Variable effective size, fixed buffer size)
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

        # =========================================================================
        # 4. Strides (Pre-calculated for speed)
        # =========================================================================
        self._stride_body = dims.num_worlds * dims.body_count
        self._stride_joint_dof = dims.num_worlds * dims.joint_dof_count
        self._stride_constraint = dims.num_worlds * dims.N_c
        self._stride_constraint_2 = dims.num_worlds * dims.N_c * 2
        self._stride_contact = dims.num_worlds * dims.contact_count
        self._stride_world = dims.num_worlds

    def zero_grad(self):
        self.body_q.grad.zero_()
        self.body_u.grad.zero_()
        self.body_f.grad.zero_()
        self.joint_target_pos.grad.zero_()
        self.joint_target_vel.grad.zero_()

    def save_step(self, step_idx: int, data: EngineData):
        """
        Saves the current state from EngineData into the buffer at step_idx.
        """
        assert step_idx >= 0, "Argument 'step_idx' has to be larger or equal to zero."

        if step_idx == 0:
            wp.copy(
                dest=self.body_q.flatten(),
                src=data.body_q_prev.flatten(),
                dest_offset=0,
                count=self._stride_body,
            )
            wp.copy(
                dest=self.body_u.flatten(),
                src=data.body_u_prev.flatten(),
                dest_offset=0,
                count=self._stride_body,
            )

        # --- Body State ---
        offset_body_curr = step_idx * self._stride_body
        offset_body_next = (step_idx + 1) * self._stride_body
        wp.copy(
            dest=self.body_q.flatten(),
            src=data.body_q.flatten(),
            dest_offset=offset_body_next,
            count=self._stride_body,
        )
        wp.copy(
            dest=self.body_u.flatten(),
            src=data.body_u.flatten(),
            dest_offset=offset_body_next,
            count=self._stride_body,
        )
        wp.copy(
            dest=self.body_f.flatten(),
            src=data.body_f.flatten(),
            dest_offset=offset_body_curr,
            count=self._stride_body,
        )

        # --- Inputs ---
        offset_dof = step_idx * self._stride_joint_dof
        wp.copy(
            dest=self.joint_target_pos.flatten(),
            src=data.joint_target_pos.flatten(),
            dest_offset=offset_dof,
            count=self._stride_joint_dof,
        )
        wp.copy(
            dest=self.joint_target_vel.flatten(),
            src=data.joint_target_vel.flatten(),
            dest_offset=offset_dof,
            count=self._stride_joint_dof,
        )

        # --- Lambdas ---
        offset_constraint_curr = step_idx * self._stride_constraint
        offset_constraint_next = (step_idx + 1) * self._stride_constraint
        offset_constraint_2_curr = step_idx * self._stride_constraint_2
        wp.copy(
            dest=self.body_lambda.flatten(),
            src=data._body_lambda.flatten(),
            dest_offset=offset_constraint_next,
            count=self._stride_constraint,
        )
        wp.copy(
            dest=self.body_lambda_prev.flatten(),
            src=data._body_lambda_prev.flatten(),
            dest_offset=offset_constraint_next,
            count=self._stride_constraint,
        )
        wp.copy(
            dest=self.constraint_active_mask.flatten(),
            src=data._constraint_active_mask.flatten(),
            dest_offset=offset_constraint_curr,
            count=self._stride_constraint,
        )
        wp.copy(
            dest=self.constraint_body_idx.flatten(),
            src=data._constraint_body_idx.flatten(),
            dest_offset=offset_constraint_2_curr,
            count=self._stride_constraint_2,
        )

        # --- Contact Manifold ---
        offset_contact_curr = step_idx * self._stride_contact
        wp.copy(
            dest=self.contact_body_a.flatten(),
            src=data.contact_body_a.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_body_b.flatten(),
            src=data.contact_body_b.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_point_a.flatten(),
            src=data.contact_point_a.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_point_b.flatten(),
            src=data.contact_point_b.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_normal.flatten(),
            src=data.contact_basis_n_a.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_dist.flatten(),
            src=data.contact_dist.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_friction.flatten(),
            src=data.contact_friction_coeff.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_restitution.flatten(),
            src=data.contact_restitution_coeff.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_thickness_a.flatten(),
            src=data.contact_thickness_a.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_thickness_b.flatten(),
            src=data.contact_thickness_b.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_basis_t1_a.flatten(),
            src=data.contact_basis_t1_a.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_basis_t2_a.flatten(),
            src=data.contact_basis_t2_a.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_basis_n_b.flatten(),
            src=data.contact_basis_n_b.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_basis_t1_b.flatten(),
            src=data.contact_basis_t1_b.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=self.contact_basis_t2_b.flatten(),
            src=data.contact_basis_t2_b.flatten(),
            dest_offset=offset_contact_curr,
            count=self._stride_contact,
        )

    def load_step(self, step_idx: int, data: EngineData):
        """
        Restores the state from the buffer into EngineData.
        """
        # --- Body State ---
        offset_body_curr = step_idx * self._stride_body
        offset_body_next = (step_idx + 1) * self._stride_body
        wp.copy(
            dest=data.body_q.flatten(),
            src=self.body_q.flatten(),
            src_offset=offset_body_next,
            count=self._stride_body,
        )
        wp.copy(
            dest=data.body_q_prev.flatten(),
            src=self.body_q.flatten(),
            src_offset=offset_body_curr,
            count=self._stride_body,
        )
        wp.copy(
            dest=data.body_u.flatten(),
            src=self.body_u.flatten(),
            src_offset=offset_body_next,
            count=self._stride_body,
        )
        wp.copy(
            dest=data.body_u_prev.flatten(),
            src=self.body_u.flatten(),
            src_offset=offset_body_curr,
            count=self._stride_body,
        )
        wp.copy(
            dest=data.body_f.flatten(),
            src=self.body_f.flatten(),
            src_offset=offset_body_curr,
            count=self._stride_body,
        )

        # --- Inputs ---
        offset_dof_curr = step_idx * self._stride_joint_dof
        wp.copy(
            dest=data.joint_target_pos.flatten(),
            src=self.joint_target_pos.flatten(),
            src_offset=offset_dof_curr,
            count=self._stride_joint_dof,
        )
        wp.copy(
            dest=data.joint_target_vel.flatten(),
            src=self.joint_target_vel.flatten(),
            src_offset=offset_dof_curr,
            count=self._stride_joint_dof,
        )

        # --- Lambdas ---
        offset_constraint_curr = step_idx * self._stride_constraint
        offset_constraint_next = (step_idx + 1) * self._stride_constraint
        offset_constraint_2_curr = step_idx * self._stride_constraint_2
        wp.copy(
            dest=data._body_lambda.flatten(),
            src=self.body_lambda.flatten(),
            src_offset=offset_constraint_next,
            count=self._stride_constraint,
        )
        wp.copy(
            dest=data._body_lambda_prev.flatten(),
            src=self.body_lambda_prev.flatten(),
            src_offset=offset_constraint_next,
            count=self._stride_constraint,
        )
        wp.copy(
            dest=data._constraint_active_mask.flatten(),
            src=self.constraint_active_mask.flatten(),
            src_offset=offset_constraint_curr,
            count=self._stride_constraint,
        )
        wp.copy(
            dest=data._constraint_body_idx.flatten(),
            src=self.constraint_body_idx.flatten(),
            src_offset=offset_constraint_2_curr,
            count=self._stride_constraint_2,
        )

        # --- Contact Manifold ---
        offset_contact_curr = step_idx * self._stride_contact
        wp.copy(
            dest=data.contact_body_a.flatten(),
            src=self.contact_body_a.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_body_b.flatten(),
            src=self.contact_body_b.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_point_a.flatten(),
            src=self.contact_point_a.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_point_b.flatten(),
            src=self.contact_point_b.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_basis_n_a.flatten(),
            src=self.contact_normal.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_dist.flatten(),
            src=self.contact_dist.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_friction_coeff.flatten(),
            src=self.contact_friction.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_restitution_coeff.flatten(),
            src=self.contact_restitution.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_thickness_a.flatten(),
            src=self.contact_thickness_a.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_thickness_b.flatten(),
            src=self.contact_thickness_b.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_basis_t1_a.flatten(),
            src=self.contact_basis_t1_a.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_basis_t2_a.flatten(),
            src=self.contact_basis_t2_a.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_basis_n_b.flatten(),
            src=self.contact_basis_n_b.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_basis_t1_b.flatten(),
            src=self.contact_basis_t1_b.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )
        wp.copy(
            dest=data.contact_basis_t2_b.flatten(),
            src=self.contact_basis_t2_b.flatten(),
            src_offset=offset_contact_curr,
            count=self._stride_contact,
        )

    def save_gradients(self, step_idx: int, data: EngineData):
        # --- Body State ---
        offset_body_curr = step_idx * self._stride_body
        wp.copy(
            dest=self.body_q.grad.flatten(),
            src=data.body_q_prev.grad.flatten(),
            dest_offset=offset_body_curr,
            count=self._stride_body,
        )
        wp.copy(
            dest=self.body_u.grad.flatten(),
            src=data.body_u_prev.grad.flatten(),
            dest_offset=offset_body_curr,
            count=self._stride_body,
        )
        wp.copy(
            dest=self.body_f.grad.flatten(),
            src=data.body_f.grad.flatten(),
            dest_offset=offset_body_curr,
            count=self._stride_body,
        )

        # --- Inputs ---
        offset_dof = step_idx * self._stride_joint_dof
        wp.copy(
            dest=self.joint_target_pos.grad.flatten(),
            src=data.joint_target_pos.grad.flatten(),
            dest_offset=offset_dof,
            count=self._stride_joint_dof,
        )
        wp.copy(
            dest=self.joint_target_vel.grad.flatten(),
            src=data.joint_target_vel.grad.flatten(),
            dest_offset=offset_dof,
            count=self._stride_joint_dof,
        )
