import warp as wp
from axion.core.contacts import AxionContacts
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions


class TrajectoryBuffer:
    def __init__(
        self,
        data: EngineData,
        contacts: AxionContacts,
        dims: EngineDimensions,
        num_steps: int,
        device,
    ):
        self.data = data
        self.contacts = contacts
        self.num_steps = num_steps
        self.dims = dims
        self.device = device

        def _alloc_buffer(
            source_array: wp.array,
            requires_grad: bool = False,
            add_one_slot: bool = False,
        ):
            if not isinstance(source_array, wp.array):
                return None

            if add_one_slot:
                dest_shape = (num_steps + 1,) + source_array.shape
            else:
                dest_shape = (num_steps,) + source_array.shape

            dest_array = wp.zeros(
                dest_shape,
                dtype=source_array.dtype,
                device=device,
                requires_grad=requires_grad,
            )
            return dest_array

        # =========================================================================
        # 1. Contact Data
        # =========================================================================
        self.target_body_pose = _alloc_buffer(data.body_pose, add_one_slot=True)
        self.target_body_vel = _alloc_buffer(data.body_vel, add_one_slot=True)

        # =========================================================================
        # 2. Body State
        # =========================================================================
        self.ext_force = _alloc_buffer(data.ext_force, True)
        self.body_pose = _alloc_buffer(data.body_pose, True, add_one_slot=True)
        self.body_vel = _alloc_buffer(data.body_vel, True, add_one_slot=True)
        self.joint_target_pos = _alloc_buffer(data.joint_target_pos, True)
        self.joint_target_vel = _alloc_buffer(data.joint_target_vel, True)

        # =========================================================================
        # 3. Constraints
        # =========================================================================
        self._constr_force = _alloc_buffer(data._constr_force)
        self._constr_force_prev_iter = _alloc_buffer(data._constr_force)

        # =========================================================================
        # 4. Contact Data
        # =========================================================================
        self.contact_count = _alloc_buffer(contacts.contact_count)
        self.contact_point0 = _alloc_buffer(contacts.contact_point0)
        self.contact_point1 = _alloc_buffer(contacts.contact_point1)
        self.contact_normal = _alloc_buffer(contacts.contact_normal)
        self.contact_shape0 = _alloc_buffer(contacts.contact_shape0)
        self.contact_shape1 = _alloc_buffer(contacts.contact_shape1)
        self.contact_thickness0 = _alloc_buffer(contacts.contact_thickness0)
        self.contact_thickness1 = _alloc_buffer(contacts.contact_thickness1)

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

    def save_target_step(self, step_idx: int, data: EngineData):
        assert step_idx >= 0, "Argument 'step_idx' has to be larger or equal to zero."

        # 1. Handle Initial Conditions (Only on first step)
        if step_idx == 0:
            wp.copy(self.target_body_pose[0], data.body_pose_prev)
            wp.copy(self.target_body_vel[0], data.body_vel_prev)

        # 2. Body State (Result of step t goes to t+1)
        wp.copy(self.target_body_pose[step_idx + 1], data.body_pose)
        wp.copy(self.target_body_vel[step_idx + 1], data.body_vel)

    def save_step(self, step_idx: int, data: EngineData, contacts: AxionContacts):
        assert step_idx >= 0, "Argument 'step_idx' has to be larger or equal to zero."

        if step_idx == 0:
            wp.copy(self.body_pose[0], data.body_pose_prev)
            wp.copy(self.body_vel[0], data.body_vel_prev)

        # 2. Body State (Result of step t goes to t+1)
        wp.copy(self.body_pose[step_idx + 1], data.body_pose)
        wp.copy(self.body_vel[step_idx + 1], data.body_vel)

        # --- Inputs ---
        wp.copy(self.ext_force[step_idx], data.ext_force)
        wp.copy(self.joint_target_pos[step_idx], data.joint_target_pos)
        wp.copy(self.joint_target_vel[step_idx], data.joint_target_vel)

        # --- Lambdas ---
        wp.copy(self._constr_force[step_idx], data._constr_force)
        wp.copy(self._constr_force_prev_iter[step_idx], data._constr_force_prev_iter)

        # --- Contacts ---
        wp.copy(self.contact_count[step_idx], contacts.contact_count)
        wp.copy(self.contact_point0[step_idx], contacts.contact_point0)
        wp.copy(self.contact_point1[step_idx], contacts.contact_point1)
        wp.copy(self.contact_normal[step_idx], contacts.contact_normal)
        wp.copy(self.contact_shape0[step_idx], contacts.contact_shape0)
        wp.copy(self.contact_shape1[step_idx], contacts.contact_shape1)
        wp.copy(self.contact_thickness0[step_idx], contacts.contact_thickness0)
        wp.copy(self.contact_thickness1[step_idx], contacts.contact_thickness1)

    def load_step(self, step_idx: int, data: EngineData, contacts: AxionContacts):
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
        wp.copy(data._constr_force, self._constr_force[step_idx])
        wp.copy(data._constr_force_prev_iter, self._constr_force_prev_iter[step_idx])

        # --- Contacts ---
        wp.copy(contacts.contact_count, self.contact_count[step_idx])
        wp.copy(contacts.contact_point0, self.contact_point0[step_idx])
        wp.copy(contacts.contact_point1, self.contact_point1[step_idx])
        wp.copy(contacts.contact_normal, self.contact_normal[step_idx])
        wp.copy(contacts.contact_shape0, self.contact_shape0[step_idx])
        wp.copy(contacts.contact_shape1, self.contact_shape1[step_idx])
        wp.copy(contacts.contact_thickness0, self.contact_thickness0[step_idx])
        wp.copy(contacts.contact_thickness1, self.contact_thickness1[step_idx])

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
