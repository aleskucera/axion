from importlib.resources import files

import hydra
import newton
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import ProfilingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

CONFIG_PATH = files("axion").joinpath("examples").joinpath("conf")


class Simulator(AbstractSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        profile_config: ProfilingConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(sim_config, render_config, exec_config, profile_config, engine_config)

    def build_model(self) -> newton.Model:
        FRICTION = 0.8
        RESTITUTION = 0.0

        builder = newton.ModelBuilder()

        # common geometry settings
        cuboid_hx = 0.1
        cuboid_hy = 0.1
        cuboid_hz = 0.75
        upper_hz = 0.25 * cuboid_hz

        # layout positions (y-rows)
        rows = [-3.0, 0.0, 3.0]
        drop_z = 2.0

        # -----------------------------
        # REVOLUTE (hinge) joint demo
        # -----------------------------
        y = rows[0]

        a_rev = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            key="a_rev",
        )
        b_rev = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz),
                q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.15),
            ),
            key="b_rev",
        )
        builder.add_shape_box(a_rev, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_rev, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        builder.add_joint_fixed(
            parent=-1,
            child=a_rev,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_revolute_anchor",
        )
        builder.add_joint_revolute(
            parent=a_rev,
            child=b_rev,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            key="revolute_a_b",
        )
        # set initial joint angle
        builder.joint_q[-1] = wp.pi * 0.5

        # -----------------------------
        # PRISMATIC (slider) joint demo
        # -----------------------------
        y = rows[1]
        a_pri = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            key="a_pri",
        )
        b_pri = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z - cuboid_hz),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.12),
            ),
            key="b_pri",
        )
        builder.add_shape_box(a_pri, hx=cuboid_hx, hy=cuboid_hy, hz=upper_hz)
        builder.add_shape_box(b_pri, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        builder.add_joint_fixed(
            parent=-1,
            child=a_pri,
            parent_xform=wp.transform(p=wp.vec3(0.0, y, drop_z + upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            key="fixed_prismatic_anchor",
        )
        builder.add_joint_prismatic(
            parent=a_pri,
            child=b_pri,
            axis=wp.vec3(0.0, 0.0, 1.0),  # slide along Z
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, -upper_hz), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            limit_lower=-0.3,
            limit_upper=0.3,
            key="prismatic_a_b",
        )

        # -----------------------------
        # BALL joint demo (sphere + cuboid)
        # -----------------------------
        y = rows[2]
        radius = 0.3
        z_offset = -1.0  # Shift down by 2 units

        # kinematic (massless) sphere as the parent anchor
        a_ball = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + cuboid_hz + z_offset), q=wp.quat_identity()
            )
        )
        b_ball = builder.add_body(
            xform=wp.transform(
                p=wp.vec3(0.0, y, drop_z + radius + z_offset),
                q=wp.quat_from_axis_angle(wp.vec3(1.0, 1.0, 0.0), 0.1),
            )
        )

        rigid_cfg = newton.ModelBuilder.ShapeConfig()
        rigid_cfg.density = 0.0
        builder.add_shape_sphere(a_ball, radius=radius, cfg=rigid_cfg)
        builder.add_shape_box(b_ball, hx=cuboid_hx, hy=cuboid_hy, hz=cuboid_hz)

        builder.add_joint_ball(
            parent=a_ball,
            child=b_ball,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, +cuboid_hz), q=wp.quat_identity()),
            key="ball_a_b",
        )
        # set initial joint angle
        builder.joint_q[-4:] = wp.quat_rpy(0.5, 0.6, 0.7)

        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=10.0, kd=10.0, kf=0.0, mu=FRICTION, restitution=RESTITUTION
            )
        )

        model = builder.finalize()
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def ball_bounce_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    profile_config: ProfilingConfig = hydra.utils.instantiate(cfg.profiling)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        profile_config=profile_config,
        engine_config=engine_config,
    )

    simulator.run()


if __name__ == "__main__":
    ball_bounce_example()
