import warp as wp


@wp.func
def compute_spatial_momentum(
    mass: wp.float32,
    inertia: wp.mat33,
    velocity: wp.spatial_vector,
) -> wp.spatial_vector:
    top = mass * wp.spatial_top(velocity)
    bot = inertia @ wp.spatial_bottom(velocity)
    return wp.spatial_vector(top, bot)


@wp.func
def compute_world_inertia(
    body_q: wp.transform,
    body_inertia: wp.mat33,
) -> wp.mat33:
    # Get orientation quaternion from transform
    orientation = wp.transform_get_rotation(body_q)
    R = wp.quat_to_matrix(orientation)

    # Transform inertia tensor: I_w = R * I_body * R^T
    I_w = R @ body_inertia @ wp.transpose(R)

    return I_w
