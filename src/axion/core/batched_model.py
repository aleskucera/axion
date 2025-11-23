import warp as wp
from newton import Contacts
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

    batched_joint_qd_start[world_idx, slot] = joint_qd_start[joint_idx] % joint_dof_count_per_world


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

    batched_shape_body[world_idx, slot] = shape_body[shape_idx] % body_count_per_world


class BatchedModel:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.device = model.device
        self.num_worlds = model.num_worlds

        self.body_count = model.body_count // model.num_worlds
        self.shape_count = model.shape_count // model.num_worlds
        self.joint_count = model.joint_count // model.num_worlds
        self.joint_dof_count = model.joint_dof_count // model.num_worlds
        self.joint_coord_count = model.joint_coord_count // model.num_worlds

        self.body_com = model.body_com.reshape((model.num_worlds, -1))
        self.body_inertia = model.body_inertia.reshape((model.num_worlds, -1))
        self.body_inv_inertia = model.body_inv_inertia.reshape((model.num_worlds, -1))
        self.body_mass = model.body_mass.reshape((model.num_worlds, -1))
        self.body_inv_mass = model.body_inv_mass.reshape((model.num_worlds, -1))

        self.joint_X_c = model.joint_X_c.reshape((model.num_worlds, -1))
        self.joint_X_p = model.joint_X_p.reshape((model.num_worlds, -1))
        self.joint_axis = model.joint_axis.reshape((model.num_worlds, -1))
        self.joint_child = model.joint_child.reshape((model.num_worlds, -1))
        self.joint_parent = model.joint_parent.reshape((model.num_worlds, -1))
        self.joint_dof_dim = model.joint_dof_dim.reshape((model.num_worlds, -1))
        self.joint_enabled = model.joint_enabled.reshape((model.num_worlds, -1))
        self.joint_target_ke = model.joint_target_ke.reshape((model.num_worlds, -1))
        self.joint_target_kd = model.joint_target_kd.reshape((model.num_worlds, -1))
        self.joint_type = model.joint_type.reshape((model.num_worlds, -1))
        self.joint_q_start = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)
        self.joint_qd_start = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)

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

        self.shape_body = wp.zeros((model.num_worlds, self.shape_count), dtype=wp.int32)
        self.shape_thickness = model.shape_thickness.reshape((model.num_worlds, -1))
        self.shape_material_mu = model.shape_material_restitution.reshape((model.num_worlds, -1))
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


@wp.kernel
def batch_contact_data_kernel(
    shape_world: wp.array(dtype=wp.int32),
    num_shapes_per_world: wp.int32,
    # Contact info
    contact_count: wp.array(dtype=wp.int32),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_thickness0: wp.array(dtype=wp.float32),
    contact_thickness1: wp.array(dtype=wp.float32),
    # OUTPUT: This array holds the assigned column index for every contact
    batched_contact_count: wp.array(dtype=wp.int32),
    batched_contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    batched_contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    batched_contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    batched_contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    batched_contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    batched_contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    batched_contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
):
    contact_idx = wp.tid()

    shape_0 = contact_shape0[contact_idx]
    shape_1 = contact_shape1[contact_idx]

    if contact_idx >= contact_count[0] or shape_0 == shape_1:
        return

    world_idx = -1
    if shape_0 >= 0:
        world_idx = shape_world[shape_0]
    if shape_1 >= 0:
        world_idx = shape_world[shape_1]

    if world_idx < 0:
        return

    slot = wp.atomic_add(batched_contact_count, world_idx, 1)

    if slot >= batched_contact_point0.shape[1]:
        return

    batched_contact_point0[world_idx, slot] = contact_point0[contact_idx]
    batched_contact_point1[world_idx, slot] = contact_point1[contact_idx]
    batched_contact_normal[world_idx, slot] = contact_normal[contact_idx]
    batched_contact_shape0[world_idx, slot] = contact_shape0[contact_idx] % num_shapes_per_world
    batched_contact_shape1[world_idx, slot] = contact_shape1[contact_idx] % num_shapes_per_world
    batched_contact_thickness0[world_idx, slot] = contact_thickness0[contact_idx]
    batched_contact_thickness1[world_idx, slot] = contact_thickness1[contact_idx]


class BatchedContacts:
    def __init__(self, model: Model, max_contacts_per_world: int) -> None:
        assert (
            model.shape_count % model.num_worlds == 0
        ), "Worlds have not identical number of shapes."

        self.model = model
        self.device = model.device
        self.num_worlds = model.num_worlds
        self.num_shapes_per_world = model.shape_count // model.num_worlds
        self.max_contacts = max_contacts_per_world

        with wp.ScopedDevice(self.device):
            self.contact_count = wp.zeros(model.num_worlds, dtype=wp.int32)
            self.contact_point0 = wp.zeros(
                (model.num_worlds, max_contacts_per_world), dtype=wp.vec3
            )
            self.contact_point1 = wp.zeros(
                (model.num_worlds, max_contacts_per_world), dtype=wp.vec3
            )
            self.contact_normal = wp.zeros(
                (model.num_worlds, max_contacts_per_world), dtype=wp.vec3
            )
            self.contact_shape0 = wp.zeros(
                (model.num_worlds, max_contacts_per_world), dtype=wp.int32
            )
            self.contact_shape1 = wp.zeros(
                (model.num_worlds, max_contacts_per_world), dtype=wp.int32
            )
            self.contact_thickness0 = wp.zeros(
                (model.num_worlds, max_contacts_per_world), dtype=wp.float32
            )
            self.contact_thickness1 = wp.zeros(
                (model.num_worlds, max_contacts_per_world), dtype=wp.float32
            )

    def load_contact_data(self, contacts: Contacts):
        self.contact_count.zero_()

        wp.launch(
            kernel=batch_contact_data_kernel,
            dim=self.model.rigid_contact_max,
            inputs=[
                self.model.shape_world,
                self.num_shapes_per_world,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
            ],
            outputs=[
                self.contact_count,
                self.contact_point0,
                self.contact_point1,
                self.contact_normal,
                self.contact_shape0,
                self.contact_shape1,
                self.contact_thickness0,
                self.contact_thickness1,
            ],
            device=self.device,
        )

        if contacts.rigid_contact_count.numpy()[0] > 0:
            print("Contaaaaaaaaaaaaaaaaaact!")
