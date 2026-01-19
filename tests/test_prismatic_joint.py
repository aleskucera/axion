import newton
import numpy as np
import pytest
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder

wp.init()


@wp.kernel
def apply_test_force_kernel(
    body_f: wp.array(dtype=wp.spatial_vector, ndim=1), force: wp.spatial_vector
):
    body_idx = wp.tid()
    # Apply force only to the dynamic body (index 0)
    if body_idx == 0:
        body_f[body_idx] = force


def compute_world_joint_frames(q_parent_np, q_child_np, parent_local_xform, child_local_xform):
    """Computes the world-space transforms of the joint frames attached to parent and child bodies."""
    if q_parent_np is None:
        t_body_parent = wp.transform_identity()
    else:
        t_body_parent = wp.transform(wp.vec3(*q_parent_np[:3]), wp.quat(*q_parent_np[3:]))

    t_body_child = wp.transform(wp.vec3(*q_child_np[:3]), wp.quat(*q_child_np[3:]))

    t_joint_parent = wp.transform_multiply(t_body_parent, parent_local_xform)
    t_joint_child = wp.transform_multiply(t_body_child, child_local_xform)

    return t_joint_parent, t_joint_child


def calculate_position_error_orthogonal(t_joint_parent, t_joint_child, axis_local):
    """Calculates the position error orthogonal to the prismatic axis."""
    p_p_wp = wp.transform_get_translation(t_joint_parent)
    p_c_wp = wp.transform_get_translation(t_joint_child)
    p_p = np.array([p_p_wp[0], p_p_wp[1], p_p_wp[2]])
    p_c = np.array([p_c_wp[0], p_c_wp[1], p_c_wp[2]])
    
    delta = p_c - p_p
    
    # Prismatic axis in world space
    q_p = wp.transform_get_rotation(t_joint_parent)
    axis_w_wp = wp.quat_rotate(q_p, axis_local)
    axis_w = np.array([axis_w_wp[0], axis_w_wp[1], axis_w_wp[2]])
    
    # Projection onto axis
    proj = np.dot(delta, axis_w)
    
    # Orthogonal part
    ortho = delta - proj * axis_w
    return np.linalg.norm(ortho), proj


def calculate_relative_angle(t_joint_parent, t_joint_child):
    """Calculates the angle of the relative rotation between two joint frames."""
    q_j0 = wp.transform_get_rotation(t_joint_parent)
    q_j1 = wp.transform_get_rotation(t_joint_child)

    q0_np = np.array([q_j0[0], q_j0[1], q_j0[2], q_j0[3]])  # x, y, z, w
    q1_np = np.array([q_j1[0], q_j1[1], q_j1[2], q_j1[3]])

    # Quaternion dot product measures similarity
    dot = abs(np.dot(q0_np, q1_np))
    # Clamp for safety
    dot = min(max(dot, -1.0), 1.0)

    # Angle = 2 * acos(|dot|)
    return 2.0 * np.arccos(dot)


def verify_prismatic(t_joint_parent, t_joint_child, axis_local):
    """Verifies Prismatic joint: Check orthogonal position alignment and relative rotation identity."""
    pos_error_ortho, sliding_dist = calculate_position_error_orthogonal(t_joint_parent, t_joint_child, axis_local)

    # Constraint Error: The relative angle should be 0
    angle_error = calculate_relative_angle(t_joint_parent, t_joint_child)

    return pos_error_ortho, angle_error, sliding_dist


def run_prismatic_test():
    # 1. Setup Model
    builder = AxionModelBuilder()

    link_1 = builder.add_link()
    builder.add_shape_box(link_1, hx=0.5, hy=0.5, hz=0.5)

    # Prismatic axis along X
    prismatic_axis = wp.vec3(1.0, 0.0, 0.0)

    # Define local transforms for the joint frames
    parent_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
    child_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())

    j0 = builder.add_joint_prismatic(
        parent=-1,
        child=link_1,
        axis=prismatic_axis,
        parent_xform=parent_local_xform,
        child_xform=child_local_xform,
    )

    builder.add_articulation([j0], key="test_articulation")

    # Use gravity along X (prismatic axis) to induce motion
    model = builder.finalize_replicated(num_worlds=1, gravity=0.0) 
    model.gravity = wp.array([wp.vec3(10.0, 0.0, 0.0)], dtype=wp.vec3, device=model.device)

    # 2. Setup Engine
    config = AxionEngineConfig(
        joint_constraint_level="pos",
        contact_constraint_level="pos",
        joint_compliance=1e-8,
        max_newton_iters=10,
        max_linear_iters=10,
    )

    def init_state_fn(state_in, state_out, contacts, dt):
        engine.integrate_bodies(model, state_in, state_out, dt)

    engine = AxionEngine(model=model, init_state_fn=init_state_fn, config=config)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.collide(state_in)
    dt = 0.01

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    for step in range(100):
        engine.step(state_in, state_out, control, contacts, dt)

        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)

        # Verification
        q_1 = state_out.body_q.numpy()[0]

        # Compute World Frames
        t_joint_0, t_joint_1 = compute_world_joint_frames(
            None, q_1, parent_local_xform, child_local_xform
        )

        pos_error_ortho, angle_error, sliding_dist = verify_prismatic(t_joint_0, t_joint_1, prismatic_axis)

    # Final checks
    assert pos_error_ortho < 1e-3, f"Prismatic Orthogonal position drift too high: {pos_error_ortho}"
    assert angle_error < 1e-3, f"Prismatic Angular error too high: {angle_error}"
    assert abs(sliding_dist) > 0.1, f"Prismatic joint did not slide significantly: {sliding_dist}"


def test_prismatic_joint():
    run_prismatic_test()


if __name__ == "__main__":
    run_prismatic_test()