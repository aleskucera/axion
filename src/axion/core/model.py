import warp as wp
from newton import Model


@wp.kernel
def batch_joint_q_start_kernel(
    joint_count_per_world: wp.int32,
    joint_coord_count_per_world: wp.int32,
    joint_q_start: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_joint_q_start: wp.array(dtype=wp.int32, ndim=2),
):
    joint_idx = wp.tid()
    if joint_idx >= joint_q_start.shape[0] - 1:
        return

    world_idx = joint_idx // joint_count_per_world
    slot = joint_idx % joint_count_per_world

    batched_joint_q_start[world_idx, slot] = joint_q_start[joint_idx] % joint_coord_count_per_world


@wp.kernel
def batch_joint_qd_start_kernel(
    joint_count_per_world: wp.int32,
    joint_dof_count_per_world: wp.int32,
    joint_qd_start: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
):
    joint_idx = wp.tid()
    if joint_idx >= joint_qd_start.shape[0] - 1:
        return

    world_idx = joint_idx // joint_count_per_world
    slot = joint_idx % joint_count_per_world

    batched_joint_qd_start[world_idx, slot] = joint_qd_start[joint_idx] % (
        joint_dof_count_per_world - 1
    )


@wp.kernel
def batch_joint_bodies_kernel(
    joint_count_per_world: wp.int32,
    body_count_per_world: wp.int32,
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_joint_parent: wp.array(dtype=wp.int32, ndim=2),
    batched_joint_child: wp.array(dtype=wp.int32, ndim=2),
):
    joint_idx = wp.tid()

    if joint_idx >= joint_parent.shape[0]:
        return

    world_idx = joint_idx // joint_count_per_world
    slot = joint_idx % joint_count_per_world

    parent_idx = joint_parent[joint_idx]
    if parent_idx >= 0:
        batched_joint_parent[world_idx, slot] = parent_idx % body_count_per_world
    else:
        batched_joint_parent[world_idx, slot] = parent_idx

    child_idx = joint_child[joint_idx]
    if child_idx >= 0:
        batched_joint_child[world_idx, slot] = child_idx % body_count_per_world
    else:
        batched_joint_child[world_idx, slot] = child_idx


@wp.kernel
def batch_shape_body_kernel(
    shape_count_per_world: wp.int32,
    body_count_per_world: wp.int32,
    shape_body: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_shape_body: wp.array(dtype=wp.int32, ndim=2),
):
    shape_idx = wp.tid()

    if shape_idx >= shape_body.shape[0]:
        return

    world_idx = shape_idx // shape_count_per_world
    slot = shape_idx % shape_count_per_world

    body_idx = shape_body[shape_idx]
    if body_idx >= 0:
        batched_shape_body[world_idx, slot] = body_idx % body_count_per_world
    else:
        batched_shape_body[world_idx, slot] = body_idx


class AxionModel:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.device = model.device
        self.num_worlds = model.num_worlds

        self.body_count = model.body_count // model.num_worlds
        self.shape_count = model.shape_count // model.num_worlds
        self.joint_count = model.joint_count // model.num_worlds
        self.joint_dof_count = model.joint_dof_count // model.num_worlds
        self.joint_coord_count = model.joint_coord_count // model.num_worlds

        with wp.ScopedDevice(self.device):
            self.body_com = model.body_com.reshape((model.num_worlds, -1))
            self.body_inertia = model.body_inertia.reshape((model.num_worlds, -1))
            self.body_inv_inertia = model.body_inv_inertia.reshape((model.num_worlds, -1))
            self.body_mass = model.body_mass.reshape((model.num_worlds, -1))
            self.body_inv_mass = model.body_inv_mass.reshape((model.num_worlds, -1))

            self.joint_X_c = model.joint_X_c.reshape((model.num_worlds, -1))
            self.joint_X_p = model.joint_X_p.reshape((model.num_worlds, -1))
            self.joint_axis = model.joint_axis.reshape((model.num_worlds, -1))
            self.joint_dof_dim = model.joint_dof_dim.reshape((model.num_worlds, -1))
            self.joint_enabled = model.joint_enabled.reshape((model.num_worlds, -1))
            self.joint_target_ke = model.joint_target_ke.reshape((model.num_worlds, -1))
            self.joint_target_kd = model.joint_target_kd.reshape((model.num_worlds, -1))
            self.joint_compliance = model.joint_compliance.reshape((model.num_worlds, -1))
            self.joint_type = model.joint_type.reshape((model.num_worlds, -1))
            self.joint_q_start = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)
            self.joint_qd_start = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)
            self.joint_parent = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)
            self.joint_child = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)

        wp.launch(
            kernel=batch_joint_q_start_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.joint_count,
                self.joint_coord_count,
                model.joint_q_start,
            ],
            outputs=[
                self.joint_q_start,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=batch_joint_qd_start_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.joint_count,
                self.joint_coord_count,
                model.joint_qd_start,
            ],
            outputs=[
                self.joint_qd_start,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=batch_joint_bodies_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.joint_count,
                self.body_count,
                model.joint_parent,
                model.joint_child,
            ],
            outputs=[
                self.joint_parent,
                self.joint_child,
            ],
            device=self.device,
        )

        with wp.ScopedDevice(self.device):
            self.shape_body = wp.zeros((model.num_worlds, self.shape_count), dtype=wp.int32)
            self.shape_thickness = model.shape_thickness.reshape((model.num_worlds, -1))
            self.shape_material_mu = model.shape_material_mu.reshape((model.num_worlds, -1))
            self.shape_material_restitution = model.shape_material_restitution.reshape(
                (model.num_worlds, -1)
            )

        wp.launch(
            kernel=batch_shape_body_kernel,
            dim=self.model.shape_count,
            inputs=[
                self.shape_count,
                self.body_count,
                model.shape_body,
            ],
            outputs=[
                self.shape_body,
            ],
            device=self.device,
        )
