import warp as wp
import newton

from axion import JointMode
from axion.core.model_builder import AxionModelBuilder

PENDULUM_HEIGHT = 5.0

def build_pendulum_model(
    num_worlds: int,
    device: wp.Device,
    requires_grad: bool = False,
) -> newton.Model:
    """Build the same 2-link revolute pendulum as examples/pendulum_AxionEngine.py,
    replicated for num_worlds."""
    builder = AxionModelBuilder()

    chain_width = 1.5
    shape_ke = 1.0e4
    shape_kd = 1.0e3
    shape_kf = 1.0e4
    hx = chain_width * 0.5

    link_config = newton.ModelBuilder.ShapeConfig(
        density=500.0, ke=shape_ke, kd=shape_kd, kf=shape_kf
    )
    capsule_xform = wp.transform(
        p=wp.vec3(0.0, 0.0, 0.0),
        q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi / 2),
    )

    link_0 = builder.add_link(armature=0.1)
    builder.add_shape_capsule(
        link_0,
        xform=capsule_xform,
        radius=0.1,
        half_height=chain_width * 0.5,
        cfg=link_config,
    )

    link_1 = builder.add_link(armature=0.1)
    builder.add_shape_capsule(
        link_1,
        xform=capsule_xform,
        radius=0.1,
        half_height=chain_width * 0.5,
        cfg=link_config,
    )

    j0 = builder.add_joint_revolute(
        parent=-1,
        child=link_0,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(
            p=wp.vec3(0.0, 0.0, PENDULUM_HEIGHT), q=wp.quat_identity()
        ),
        child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        target_ke=1000.0,
        target_kd=50.0,
        custom_attributes={
            #"joint_target_ki": [0.5],
            "joint_dof_mode": [JointMode.NONE],
        },
    )
    j1 = builder.add_joint_revolute(
        parent=link_0,
        child=link_1,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
        target_ke=500.0,
        target_kd=5.0,
        custom_attributes={
            #"joint_target_ki": [0.5],
            "joint_dof_mode": [JointMode.NONE],
        },
        armature=0.1,
    )

    builder.add_articulation([j0, j1], key="pendulum")
    builder.add_ground_plane()

    return builder.finalize_replicated(
        num_worlds=num_worlds,
        gravity=-9.81,
        device=device,
        requires_grad=requires_grad,
    )