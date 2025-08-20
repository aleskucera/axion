import warp as wp


@wp.struct
class GeneralizedMass:
    m: wp.float32
    inertia: wp.mat33


@wp.func
def add(
    a: GeneralizedMass,
    b: GeneralizedMass,
) -> GeneralizedMass:
    return GeneralizedMass(a.m + b.m, a.inertia + b.inertia)


@wp.func
def mul(
    a: GeneralizedMass,
    b: wp.spatial_vector,
) -> wp.spatial_vector:
    top = a.inertia @ wp.spatial_top(b)
    bot = a.m * wp.spatial_bottom(b)
    return wp.spatial_vector(top, bot)


@wp.func
def mul(
    a: wp.spatial_vector,
    b: GeneralizedMass,
) -> wp.spatial_vector:
    top = wp.spatial_top(a) @ b.inertia
    bot = wp.spatial_bottom(a) * b.m
    return wp.spatial_vector(top, bot)


@wp.func
def scale(M: GeneralizedMass, s: wp.float32) -> GeneralizedMass:
    """Scale generalized mass by scalar."""
    return GeneralizedMass(M.m * s, M.inertia * s)


@wp.kernel
def generalized_mass_kernel(
    mass: wp.array(dtype=wp.float32),
    inertia: wp.array(dtype=wp.mat33),
    # Outputs
    generalized_mass: wp.array(dtype=GeneralizedMass),
):
    tid = wp.tid()

    M = GeneralizedMass(mass[tid], inertia[tid])
    generalized_mass[tid] = M
