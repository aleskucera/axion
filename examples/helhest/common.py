import pathlib
from typing import Dict
from typing import List
from typing import Tuple

import newton
import numpy as np
import openmesh
import warp as wp
from axion import JointMode

ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


class HelhestConfig:
    """Configuration for the Helhest robot model."""

    # Chassis
    CHASSIS_SIZE = [0.26, 0.6, 0.18]
    CHASSIS_MASS = 19.0
    CHASSIS_OFFSET = wp.vec3(-0.047, 0.0, 0.0)
    CHASSIS_I = wp.mat33(0.6213, 0.0, 0.0, 0.0, 0.1583, 0.0, 0.0, 0.0, 0.6770)

    # Wheels
    WHEEL_RADIUS = 0.36
    WHEEL_WIDTH = 0.11
    WHEEL_MASS = 6.0
    WHEEL_I = wp.mat33(0.20045, 0.0, 0.0, 0.0, 0.20045, 0.0, 0.0, 0.0, 0.3888)
    WHEEL_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0)

    # Wheel Positions
    LEFT_WHEEL_POS = wp.vec3(0.0, 0.36, 0.0)
    RIGHT_WHEEL_POS = wp.vec3(0.0, -0.36, 0.0)
    REAR_WHEEL_POS = wp.vec3(-0.697, 0.0, 0.0)

    # Joint Control
    TARGET_KE = 50.0
    TARGET_KI = 0.04
    TARGET_KD = 0.04

    # Fixed Components Configuration
    # Format: name: (position, size, mass, inertia_diagonal)
    FIXED_COMPONENTS = {
        "battery": (
            wp.vec3(-0.302, 0.165, 0.0),
            [0.25, 0.1, 0.19],
            2.0,
            [0.00768, 0.0164, 0.01208],
        ),
        "left_motor": (
            wp.vec3(-0.09, 0.14, 0.0),
            [0.085, 0.24, 0.085],
            7.0,
            [0.0378, 0.0084, 0.0378],
        ),
        "right_motor": (
            wp.vec3(-0.09, -0.14, 0.0),
            [0.085, 0.24, 0.085],
            7.0,
            [0.0378, 0.0084, 0.0378],
        ),
        "rear_motor": (
            wp.vec3(-0.22, -0.04, 0.0),
            [0.085, 0.24, 0.085],
            7.0,
            [0.0378, 0.0084, 0.0378],
        ),
        "left_wheel_holder": (
            wp.vec3(-0.477, 0.095, 0.0),
            [0.6, 0.04, 0.18],
            3.0,
            [0.0085, 0.0981, 0.0904],
        ),
        "right_wheel_holder": (
            wp.vec3(-0.477, -0.095, 0.0),
            [0.6, 0.04, 0.18],
            3.0,
            [0.0085, 0.0981, 0.0904],
        ),
    }


def _load_wheel_mesh():
    """Loads and prepares the wheel mesh."""
    wheel_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("helhest/wheel2.obj")))
    scale = np.array([0.8, 0.8, 0.8])
    mesh_points = np.array(wheel_m.points()) * scale
    mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
    return newton.Mesh(mesh_points, mesh_indices)


def _add_chassis(builder: newton.ModelBuilder, xform: wp.transform, is_visible: bool) -> int:
    """Adds the chassis link and shape."""
    chassis = builder.add_link(
        xform=xform,
        key="chassis",
        mass=HelhestConfig.CHASSIS_MASS,
        I_m=HelhestConfig.CHASSIS_I,
        com=HelhestConfig.CHASSIS_OFFSET,
    )

    builder.add_shape_box(
        body=chassis,
        xform=wp.transform(HelhestConfig.CHASSIS_OFFSET, wp.quat_identity()),
        hx=HelhestConfig.CHASSIS_SIZE[0] / 2.0,
        hy=HelhestConfig.CHASSIS_SIZE[1] / 2.0,
        hz=HelhestConfig.CHASSIS_SIZE[2] / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            is_visible=is_visible,
            collision_group=-1,
        ),
    )
    return chassis


def _add_wheel(
    builder: newton.ModelBuilder,
    parent_xform: wp.transform,
    name: str,
    pos_local: wp.vec3,
    mu: float,
    wheel_mesh: newton.Mesh,
    is_visible: bool,
    mesh_rotation: wp.quat = wp.quat_identity(),
) -> int:
    """Adds a wheel link, shapes, and returns the link index."""
    pos_world = wp.transform_point(parent_xform, pos_local)
    rot_world = parent_xform.q

    wheel_link = builder.add_link(
        xform=wp.transform(pos_world, rot_world),
        key=name,
        mass=HelhestConfig.WHEEL_MASS,
        I_m=HelhestConfig.WHEEL_I,
        com=None,
    )

    # Visual Mesh
    builder.add_shape_mesh(
        body=wheel_link,
        mesh=wheel_mesh,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), mesh_rotation),
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            collision_group=0,
            is_visible=is_visible,
        ),
    )

    # Collision Shape
    builder.add_shape_capsule(
        body=wheel_link,
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), HelhestConfig.WHEEL_ROT),
        radius=HelhestConfig.WHEEL_RADIUS,
        half_height=HelhestConfig.WHEEL_WIDTH / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            is_visible=False,
            collision_group=-1,
            mu=mu,
            contact_margin=0.3,
        ),
    )
    return wheel_link


def _add_fixed_component(
    builder: newton.ModelBuilder,
    parent_body: int,
    parent_xform: wp.transform,
    name: str,
    params: Tuple,
    is_visible: bool,
) -> int:
    """Adds a fixed component and its joint to the chassis."""
    pos, size, mass, inertia_diag = params

    # Calculate world transform
    pos_world = wp.transform_point(parent_xform, pos)
    rot_world = parent_xform.q

    inertia_tensor = wp.mat33(
        inertia_diag[0], 0.0, 0.0, 0.0, inertia_diag[1], 0.0, 0.0, 0.0, inertia_diag[2]
    )

    link = builder.add_link(
        xform=wp.transform(pos_world, rot_world),
        key=name,
        mass=mass,
        I_m=inertia_tensor,
        com=wp.vec3(0.0, 0.0, 0.0),
    )

    builder.add_shape_box(
        body=link,
        xform=wp.transform_identity(),
        hx=size[0] / 2.0,
        hy=size[1] / 2.0,
        hz=size[2] / 2.0,
        cfg=newton.ModelBuilder.ShapeConfig(
            density=0.0,
            is_visible=is_visible,
            collision_group=0,
        ),
    )

    joint = builder.add_joint_fixed(
        parent=parent_body,
        child=link,
        parent_xform=wp.transform(pos, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        key=f"{name}_j",
    )

    return joint


def create_helhest_model(
    builder: newton.ModelBuilder,
    xform: wp.transform = wp.transform_identity(),
    is_visible: bool = True,
):
    """
    Creates a Helhest robot model using physical parameters from the URDF
    and visual assets from the examples.

    Args:
        builder: The model builder to add the robot to.
        xform: The world transform of the robot base.
        is_visible: Whether to enable visual shapes.
    """

    wheel_mesh_render = _load_wheel_mesh()

    # 1. Chassis
    chassis = _add_chassis(builder, xform, is_visible)
    j_base = builder.add_joint_free(parent=-1, child=chassis, key="base_joint")

    # 2. Wheels
    left_wheel = _add_wheel(
        builder,
        xform,
        "left_wheel",
        HelhestConfig.LEFT_WHEEL_POS,
        1.0,
        wheel_mesh_render,
        is_visible,
    )
    right_wheel = _add_wheel(
        builder,
        xform,
        "right_wheel",
        HelhestConfig.RIGHT_WHEEL_POS,
        1.0,
        wheel_mesh_render,
        is_visible,
    )
    rear_wheel = _add_wheel(
        builder,
        xform,
        "rear_wheel",
        HelhestConfig.REAR_WHEEL_POS,
        1.0,
        wheel_mesh_render,
        is_visible,
    )

    # 3. Wheel Joints
    Y_AXIS = (0.0, 1.0, 0.0)

    j_left = builder.add_joint_revolute(
        parent=chassis,
        child=left_wheel,
        parent_xform=wp.transform(HelhestConfig.LEFT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=HelhestConfig.TARGET_KE,
        target_kd=HelhestConfig.TARGET_KD,
        key="left_wheel_j",
        custom_attributes={
            "joint_target_ki": [HelhestConfig.TARGET_KI],
            "joint_dof_mode": [JointMode.TARGET_VELOCITY],
        },
    )

    j_right = builder.add_joint_revolute(
        parent=chassis,
        child=right_wheel,
        parent_xform=wp.transform(HelhestConfig.RIGHT_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=HelhestConfig.TARGET_KE,
        target_kd=HelhestConfig.TARGET_KD,
        key="right_wheel_j",
        custom_attributes={
            "joint_target_ki": [HelhestConfig.TARGET_KI],
            "joint_dof_mode": [JointMode.TARGET_VELOCITY],
        },
    )

    j_rear = builder.add_joint_revolute(
        parent=chassis,
        child=rear_wheel,
        parent_xform=wp.transform(HelhestConfig.REAR_WHEEL_POS, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=Y_AXIS,
        target_ke=HelhestConfig.TARGET_KE,
        target_kd=HelhestConfig.TARGET_KD,
        key="rear_wheel_j",
        custom_attributes={
            "joint_target_ki": [HelhestConfig.TARGET_KI],
            "joint_dof_mode": [JointMode.TARGET_VELOCITY],
        },
    )

    # 4. Fixed Components
    fixed_joints = []
    for name, params in HelhestConfig.FIXED_COMPONENTS.items():
        j_fixed = _add_fixed_component(builder, chassis, xform, name, params, is_visible)
        fixed_joints.append(j_fixed)

    # 5. Articulation
    builder.add_articulation([j_base, j_left, j_right, j_rear] + fixed_joints, key="helhest")

    return chassis, [left_wheel]
