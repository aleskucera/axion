import warp as wp
from newton.geometry import transform_inertia


@wp.struct
class SpatialInertia:
    m: wp.float32
    inertia: wp.mat33


@wp.func
def add_inertia(
    a: SpatialInertia,
    b: SpatialInertia,
) -> SpatialInertia:
    return SpatialInertia(a.m + b.m, a.inertia + b.inertia)


@wp.func
def to_spatial_momentum(
    a: SpatialInertia,
    b: wp.spatial_vector,
) -> wp.spatial_vector:
    top = a.m * wp.spatial_top(b)
    bot = a.inertia @ wp.spatial_bottom(b)
    return wp.spatial_vector(top, bot)


@wp.kernel
def spatial_inertia_kernel(
    mass: wp.array(dtype=wp.float32),
    inertia: wp.array(dtype=wp.mat33),
    # Outputs
    spatial_inertia: wp.array(dtype=SpatialInertia),
):
    tid = wp.tid()
    spatial_inertia[tid] = SpatialInertia(mass[tid], inertia[tid])


@wp.kernel
def world_spatial_inertia_kernel(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33, ndim=2),
    # Outputs
    world_spatial_inertia: wp.array(dtype=SpatialInertia, ndim=2),
):
    world_idx, body_idx = wp.tid()

    # Get body transform and spatial inertia
    transform = body_q[world_idx, body_idx]
    m = body_mass[world_idx, body_idx]
    I = body_inertia[world_idx, body_idx]

    # Get orientation quaternion from transform
    orientation = wp.transform_get_rotation(transform)

    R = wp.quat_to_matrix(orientation)
    I_w = R @ I @ wp.transpose(R)

    # Store the result
    world_spatial_inertia[world_idx, body_idx] = SpatialInertia(m, I_w)
