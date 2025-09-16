import warp as wp


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
    top = a.inertia @ wp.spatial_top(b)
    bot = a.m * wp.spatial_bottom(b)
    return wp.spatial_vector(top, bot)


@wp.kernel
def assemble_spatial_inertia_kernel(
    mass: wp.array(dtype=wp.float32),
    inertia: wp.array(dtype=wp.mat33),
    # Outputs
    spatial_inertia: wp.array(dtype=SpatialInertia),
):
    tid = wp.tid()

    M = SpatialInertia(mass[tid], inertia[tid])
    spatial_inertia[tid] = M
