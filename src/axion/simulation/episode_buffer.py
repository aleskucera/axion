import warp as wp
from newton import Control
from newton import Model
from newton import State


@wp.kernel
def save_state_and_increment_idx_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_u: wp.array(dtype=wp.spatial_vector),
    state_idx: wp.array(dtype=wp.int32),
    body_q_buffer: wp.array(dtype=wp.transform, ndim=2),
    body_u_buffer: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    body_idx = wp.tid()
    if body_idx >= body_q.shape[0]:
        return

    body_q_buffer[state_idx[0], body_idx] = body_q[body_idx]
    body_u_buffer[state_idx[0], body_idx] = body_u[body_idx]

    if body_idx > 0:  # Make sure that we increment only once
        return

    state_idx[0] = state_idx[0] + 1


@wp.kernel
def save_state_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_u: wp.array(dtype=wp.spatial_vector),
    state_idx: wp.int32,
    body_q_buffer: wp.array(dtype=wp.transform, ndim=2),
    body_u_buffer: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    body_idx = wp.tid()
    if body_idx >= body_q.shape[0]:
        return

    body_q_buffer[state_idx, body_idx] = body_q[body_idx]
    body_u_buffer[state_idx, body_idx] = body_u[body_idx]


@wp.kernel
def apply_external_force_and_increment_idx_kernel(
    body_f_buffer: wp.array(dtype=wp.spatial_vector, ndim=2),
    external_force_idx: wp.array(dtype=wp.int32),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    body_idx = wp.tid()
    if body_idx >= body_f.shape[0]:
        return

    body_f[body_idx] = body_f_buffer[external_force_idx[0], body_idx]

    if body_idx > 0:  # Make sure that we increment only once
        return

    external_force_idx[0] = external_force_idx[0] + 1


@wp.kernel
def apply_external_force_kernel(
    body_f_buffer: wp.array(dtype=wp.spatial_vector, ndim=2),
    external_force_idx: wp.int32,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    body_idx = wp.tid()
    if body_idx >= body_f.shape[0]:
        return

    body_f[body_idx] = body_f_buffer[external_force_idx, body_idx]


@wp.kernel
def apply_control_and_increment_idx_kernel(
    joint_f_buffer: wp.array(dtype=wp.float32, ndim=2),
    joint_target_pos_buffer: wp.array(dtype=wp.float32, ndim=2),
    joint_target_vel_buffer: wp.array(dtype=wp.float32, ndim=2),
    control_idx: wp.array(dtype=wp.int32),
    joint_f: wp.array(dtype=wp.float32),
    joint_target_pos: wp.array(dtype=wp.float32),
    joint_target_vel: wp.array(dtype=wp.float32),
):
    joint_dof_idx = wp.tid()
    if joint_dof_idx >= joint_target_vel.shape[0]:
        return

    joint_f[joint_dof_idx] = joint_f_buffer[control_idx[0], joint_dof_idx]
    joint_target_pos[joint_dof_idx] = joint_target_pos_buffer[control_idx[0], joint_dof_idx]
    joint_target_vel[joint_dof_idx] = joint_target_vel_buffer[control_idx[0], joint_dof_idx]

    if joint_dof_idx > 0:  # Make sure that we increment only once
        return

    control_idx[0] = control_idx[0] + 1


@wp.kernel
def apply_control_kernel(
    joint_f_buffer: wp.array(dtype=wp.float32, ndim=2),
    joint_target_pos_buffer: wp.array(dtype=wp.float32, ndim=2),
    joint_target_vel_buffer: wp.array(dtype=wp.float32, ndim=2),
    control_idx: wp.int32,
    joint_f: wp.array(dtype=wp.float32),
    joint_target_pos: wp.array(dtype=wp.float32),
    joint_target_vel: wp.array(dtype=wp.float32),
):
    joint_dof_idx = wp.tid()
    if joint_dof_idx >= joint_target_vel.shape[0]:
        return

    joint_f[joint_dof_idx] = joint_f_buffer[control_idx, joint_dof_idx]
    joint_target_pos[joint_dof_idx] = joint_target_pos_buffer[control_idx, joint_dof_idx]
    joint_target_vel[joint_dof_idx] = joint_target_vel_buffer[control_idx, joint_dof_idx]


class EpisodeBuffer:
    def __init__(self, model: Model, num_steps: int, device: wp.Device) -> None:
        self.model = model
        self.num_steps = num_steps

        self.device = device

        with wp.ScopedDevice(self.device):

            # --- STATE ---
            self.state_idx = wp.zeros(1, dtype=wp.int32)
            self.body_q = wp.zeros(
                shape=(self.num_steps + 1, self.model.body_count),
                dtype=wp.transform,
                requires_grad=True,
            )
            self.body_qd = wp.zeros(
                shape=(self.num_steps + 1, self.model.body_count),
                dtype=wp.spatial_vector,
                requires_grad=True,
            )

            # --- EXTERNAL FORCE ---
            self.external_force_idx = wp.zeros(1, dtype=wp.int32)
            self.body_f = wp.full(
                value=wp.spatial_vector(0.0, 1e2, -1e2, 0.0, 0.0, 0.0),
                shape=(self.num_steps, self.model.body_count),
                dtype=wp.spatial_vector,
                requires_grad=True,
            )

            # --- CONTROL ---
            self.control_idx = wp.zeros(1, dtype=wp.int32)
            self.joint_f = wp.zeros(
                shape=(self.num_steps, self.model.joint_dof_count),
                dtype=wp.float32,
                requires_grad=True,
            )
            self.joint_target_pos = wp.zeros(
                shape=(self.num_steps, self.model.joint_dof_count),
                dtype=wp.float32,
                requires_grad=True,
            )
            self.joint_target_vel = wp.zeros(
                shape=(self.num_steps, self.model.joint_dof_count),
                dtype=wp.float32,
                requires_grad=True,
            )

    def reset(self):
        self.state_idx.zero_()
        # self.body_q.zero_()
        # self.body_q.grad.zero_()
        # self.body_qd.zero_()
        # self.body_qd.grad.zero_()

        self.external_force_idx.zero_()
        # self.body_f.zero_()
        self.body_f.grad.zero_()

        self.control_idx.zero_()
        # self.body_joint_target_pos.zero_()
        self.joint_target_pos.grad.zero_()
        # self.body_joint_target_vel.zero_()
        self.joint_target_vel.grad.zero_()

    def save_state_and_increment_idx(self, state: State):
        wp.launch(
            kernel=save_state_and_increment_idx_kernel,
            dim=self.model.body_count,
            inputs=[
                state.body_q,
                state.body_qd,
            ],
            outputs=[
                self.state_idx,
                self.body_q,
                self.body_qd,
            ],
            device=self.device,
        )

    def save_state(self, state: State, state_idx: int):
        wp.launch(
            kernel=save_state_kernel,
            dim=self.model.body_count,
            inputs=[
                state.body_q,
                state.body_qd,
                state_idx,
            ],
            outputs=[
                self.body_q,
                self.body_qd,
            ],
            device=self.device,
        )

    def apply_external_force_and_increment_idx(self, state: State):
        wp.launch(
            kernel=apply_external_force_and_increment_idx_kernel,
            dim=self.model.body_count,
            inputs=[
                self.body_f,
            ],
            outputs=[
                self.external_force_idx,
                state.body_f,
            ],
            device=self.device,
        )

    def apply_external_force(self, state: State, external_force_idx: int):
        wp.launch(
            kernel=apply_external_force_kernel,
            dim=self.model.body_count,
            inputs=[
                self.body_f,
                external_force_idx,
            ],
            outputs=[
                state.body_f,
            ],
            device=self.device,
        )

    def apply_control_and_increment_idx(self, control: Control):
        wp.launch(
            kernel=apply_control_and_increment_idx_kernel,
            dim=self.model.joint_dof_count,
            inputs=[
                self.joint_f,
                self.joint_target_pos,
                self.joint_target_vel,
            ],
            outputs=[
                self.control_idx,
                control.joint_f,
                control.joint_target_pos,
                control.joint_target_vel,
            ],
            device=self.device,
        )

    def apply_control(self, control: Control, control_idx: int):
        wp.launch(
            kernel=apply_control_kernel,
            dim=self.model.joint_dof_count,
            inputs=[
                self.joint_f,
                self.joint_target_pos,
                self.joint_target_vel,
                control_idx,
            ],
            outputs=[
                control.joint_f,
                control.joint_target_pos,
                control.joint_target_vel,
            ],
            device=self.device,
        )
