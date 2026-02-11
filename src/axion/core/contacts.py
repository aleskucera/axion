import warp as wp
from axion.constraints import fill_contact_constraint_body_idx_kernel
from axion.constraints import fill_friction_constraint_body_idx_kernel
from axion.math import orthogonal_basis
from newton import Contacts
from newton import Model

from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel


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


@wp.kernel
def contact_interaction_kernel(
    # --- Inputs ---
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_com: wp.array(dtype=wp.vec3, ndim=2),
    shape_body: wp.array(dtype=wp.int32, ndim=2),
    shape_thickness: wp.array(dtype=wp.float32, ndim=2),
    shape_material_mu: wp.array(dtype=wp.float32, ndim=2),
    shape_material_restitution: wp.array(dtype=wp.float32, ndim=2),
    contact_count: wp.array(dtype=wp.int32, ndim=1),
    contact_point0: wp.array(dtype=wp.vec3, ndim=2),
    contact_point1: wp.array(dtype=wp.vec3, ndim=2),
    contact_normal: wp.array(dtype=wp.vec3, ndim=2),
    contact_shape0: wp.array(dtype=wp.int32, ndim=2),
    contact_shape1: wp.array(dtype=wp.int32, ndim=2),
    contact_thickness0: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness1: wp.array(dtype=wp.float32, ndim=2),
    # --- Outputs ---
    contact_body_a: wp.array(dtype=wp.int32, ndim=2),
    contact_body_b: wp.array(dtype=wp.int32, ndim=2),
    contact_point_a: wp.array(dtype=wp.vec3, ndim=2),
    contact_point_b: wp.array(dtype=wp.vec3, ndim=2),
    contact_thickness_a: wp.array(dtype=wp.float32, ndim=2),
    contact_thickness_b: wp.array(dtype=wp.float32, ndim=2),
    contact_dist: wp.array(dtype=wp.float32, ndim=2),
    contact_friction_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_restitution_coeff: wp.array(dtype=wp.float32, ndim=2),
    contact_basis_n_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t1_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_a: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_n_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t1_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_basis_t2_b: wp.array(dtype=wp.spatial_vector, ndim=2),
    contact_active_mask: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, contact_idx = wp.tid()

    shape_a = contact_shape0[world_idx, contact_idx]
    shape_b = contact_shape1[world_idx, contact_idx]

    if contact_idx >= contact_count[world_idx] or shape_a == shape_b:
        contact_active_mask[world_idx, contact_idx] = 0.0
        return

    contact_active_mask[world_idx, contact_idx] = 1.0

    # Contact body indices (default to -1)
    body_a = -1
    body_b = -1

    # Get body indices
    if shape_a >= 0:
        body_a = shape_body[world_idx, shape_a]
    if shape_b >= 0:
        body_b = shape_body[world_idx, shape_b]

    # Contact normal in world space
    n = contact_normal[world_idx, contact_idx]

    # Contact points from Newton are in Body-Local space
    p_a_local = contact_point0[world_idx, contact_idx]
    p_b_local = contact_point1[world_idx, contact_idx]

    # Transforms
    X_a = wp.transform_identity()
    if body_a >= 0:
        X_a = body_q[world_idx, body_a]

    X_b = wp.transform_identity()
    if body_b >= 0:
        X_b = body_q[world_idx, body_b]

    # World points
    p_a_world = wp.transform_point(X_a, p_a_local)
    p_b_world = wp.transform_point(X_b, p_b_local)

    # Thickness adjustment in world space
    offset_a = -contact_thickness0[world_idx, contact_idx] * n
    offset_b = contact_thickness1[world_idx, contact_idx] * n

    p_a_world_adj = p_a_world + offset_a
    p_b_world_adj = p_b_world + offset_b

    # Contact residuals (lever arms) in World space
    r_a = wp.vec3()
    if body_a >= 0:
        com_a_world = wp.transform_point(X_a, body_com[world_idx, body_a])
        r_a = p_a_world_adj - com_a_world

    r_b = wp.vec3()
    if body_b >= 0:
        com_b_world = wp.transform_point(X_b, body_com[world_idx, body_b])
        r_b = p_b_world_adj - com_b_world

    # Compute contact Jacobians
    t1, t2 = orthogonal_basis(n)

    # Fill the output arrays
    contact_body_a[world_idx, contact_idx] = body_a
    contact_body_b[world_idx, contact_idx] = body_b

    contact_point_a[world_idx, contact_idx] = p_a_local
    contact_point_b[world_idx, contact_idx] = p_b_local

    contact_thickness_a[world_idx, contact_idx] = contact_thickness0[world_idx, contact_idx]
    contact_thickness_b[world_idx, contact_idx] = contact_thickness1[world_idx, contact_idx]

    # Penetration depth (positive for penetration)
    contact_dist[world_idx, contact_idx] = wp.dot(n, p_b_world_adj - p_a_world_adj)

    contact_basis_n_a[world_idx, contact_idx] = wp.spatial_vector(n, wp.cross(r_a, n))
    contact_basis_t1_a[world_idx, contact_idx] = wp.spatial_vector(t1, wp.cross(r_a, t1))
    contact_basis_t2_a[world_idx, contact_idx] = wp.spatial_vector(t2, wp.cross(r_a, t2))

    contact_basis_n_b[world_idx, contact_idx] = wp.spatial_vector(-n, -wp.cross(r_b, n))
    contact_basis_t1_b[world_idx, contact_idx] = wp.spatial_vector(-t1, -wp.cross(r_b, t1))
    contact_basis_t2_b[world_idx, contact_idx] = wp.spatial_vector(-t2, -wp.cross(r_b, t2))

    mu_a = shape_material_mu[world_idx, shape_a]
    mu_b = shape_material_mu[world_idx, shape_b]
    contact_friction_coeff[world_idx, contact_idx] = (mu_a + mu_b) * 0.5

    e_a = shape_material_restitution[world_idx, shape_a]
    e_b = shape_material_restitution[world_idx, shape_b]
    contact_restitution_coeff[world_idx, contact_idx] = (e_a + e_b) * 0.5


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

    def load_contact_data(
        self, contacts: Contacts, axion_model: AxionModel, data: EngineData, dims: EngineDimensions
    ):
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

        wp.launch(
            kernel=contact_interaction_kernel,
            dim=(axion_model.num_worlds, dims.contact_count),
            inputs=[
                data.body_pose,
                axion_model.body_com,
                axion_model.shape_body,
                axion_model.shape_thickness,
                axion_model.shape_material_mu,
                axion_model.shape_material_restitution,
                self.contact_count,
                self.contact_point0,
                self.contact_point1,
                self.contact_normal,
                self.contact_shape0,
                self.contact_shape1,
                self.contact_thickness0,
                self.contact_thickness1,
            ],
            outputs=[
                data.contact_body_a,
                data.contact_body_b,
                data.contact_point_a,
                data.contact_point_b,
                data.contact_thickness_a,
                data.contact_thickness_b,
                data.contact_dist,
                data.contact_friction_coeff,
                data.contact_restitution_coeff,
                data.contact_basis_n_a,
                data.contact_basis_t1_a,
                data.contact_basis_t2_a,
                data.contact_basis_n_b,
                data.contact_basis_t1_b,
                data.contact_basis_t2_b,
                data.constr_active_mask.n,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_contact_constraint_body_idx_kernel,
            dim=(axion_model.num_worlds, dims.contact_count),
            inputs=[
                data.contact_body_a,
                data.contact_body_b,
            ],
            outputs=[
                data.constr_body_idx.n,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=fill_friction_constraint_body_idx_kernel,
            dim=(axion_model.num_worlds, 2 * dims.contact_count),
            inputs=[
                data.contact_body_a,
                data.contact_body_b,
            ],
            outputs=[
                data.constr_body_idx.f,
            ],
            device=self.device,
        )
