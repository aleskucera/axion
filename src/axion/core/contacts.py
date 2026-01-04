import warp as wp
from newton import Contacts
from newton import Model


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
    batched_contact_thickness0[world_idx, slot] = contact_thickness0[contact_idx]
    batched_contact_thickness1[world_idx, slot] = contact_thickness1[contact_idx]

    if shape_0 >= 0:
        batched_contact_shape0[world_idx, slot] = shape_0 % num_shapes_per_world
    else:
        batched_contact_shape0[world_idx, slot] = shape_0

    if shape_1 >= 0:
        batched_contact_shape1[world_idx, slot] = shape_1 % num_shapes_per_world
    else:
        batched_contact_shape1[world_idx, slot] = shape_1


class AxionContacts:
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

        # if contacts.rigid_contact_count.numpy()[0] > 0:
        #     print("Contact")
