import time

import numpy as np
import warp as wp
from axion.ncp import scaled_fisher_burmeister
from warp.sim import ModelShapeGeometry
from warp.sim import ModelShapeMaterials


@wp.func
def _compute_friction_coefficient(
    shape_a: wp.int32,
    shape_b: wp.int32,
    shape_materials: ModelShapeMaterials,
) -> wp.float32:
    mu = 0.0
    if shape_a >= 0 and shape_b >= 0:
        mu_a = shape_materials.mu[shape_a]
        mu_b = shape_materials.mu[shape_b]
        mu = (mu_a + mu_b) * 0.5
    elif shape_a >= 0:
        mu = shape_materials.mu[shape_a]
    elif shape_b >= 0:
        mu = shape_materials.mu[shape_b]
    return mu


@wp.func
def _compute_complementarity_argument(
    grad_c_n_a: wp.spatial_vector,
    grad_c_n_b: wp.spatial_vector,
    body_qd_a: wp.spatial_vector,
    body_qd_b: wp.spatial_vector,
    body_qd_prev_a: wp.spatial_vector,
    body_qd_prev_b: wp.spatial_vector,
    c_n: wp.float32,
    restitution: wp.float32,
    dt: wp.float32,
    stabilization_factor: wp.float32,
) -> wp.float32:
    # Relative normal velocity at the current time step
    delta_v_n = wp.dot(grad_c_n_a, body_qd_a) + wp.dot(grad_c_n_b, body_qd_b)

    # Relative normal velocity at the previous time step
    delta_v_n_prev = wp.dot(grad_c_n_a, body_qd_prev_a) + wp.dot(
        grad_c_n_b, body_qd_prev_b
    )

    # Baumgarte stabilization bias from penetration depth
    b_err = stabilization_factor / dt * c_n

    # Restitution bias from previous velocity
    b_rest = -restitution * delta_v_n_prev

    return delta_v_n + b_err + b_rest
