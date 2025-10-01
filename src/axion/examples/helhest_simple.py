from importlib.resources import files

import hydra
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

        self.control.joint_act = wp.array(
            [
                3.0,  # left wheel target velocity
                3.0,  # right wheel target velocity
                0.0,  # back wheel (force control, so velocity is 0)
            ],
            dtype=wp.float32,
        )

        # Set the stiffness (motor strength) for the velocity-controlled joints.
        self.model.joint_target_ke = wp.array(
            [
                500.0,  # left wheel
                500.0,  # right wheel
                0.0,  # back wheel
            ],
            dtype=wp.float32,
        )

    def build_model(self) -> wp.sim.Model:
        FRICTION = 0.9
        RESTITUTION = 0.0

        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # --- Build the Vehicle ---

        # Create main body (chassis)
        chassis = builder.add_body(
            origin=wp.transform((0.0, 0.0, 1.2), wp.quat_identity()), name="chassis"
        )
        builder.add_shape_box(
            body=chassis,
            hx=1.5,
            hy=0.5,
            hz=0.5,
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
        )

        # Left Wheel
        left_wheel = builder.add_body(
            origin=wp.transform((1.5, -1.5, 1.2), wp.quat_identity()), name="left_wheel"
        )
        builder.add_shape_sphere(
            body=left_wheel,
            radius=1.0,
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.1,  # Visual flair
        )

        # Right Wheel
        right_wheel = builder.add_body(
            origin=wp.transform((1.5, 1.5, 1.2), wp.quat_identity()), name="right_wheel"
        )
        builder.add_shape_sphere(
            body=right_wheel,
            radius=1.0,
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
        )

        # Back Wheel
        back_wheel = builder.add_body(
            origin=wp.transform((-2.5, 0.0, 1.2), wp.quat_identity()), name="back_wheel"
        )
        builder.add_shape_sphere(
            body=back_wheel,
            radius=1.0,
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
        )

        # --- Define Joints ---

        # Left wheel revolute joint (velocity control)
        builder.add_joint_revolute(
            parent=chassis,
            child=left_wheel,
            parent_xform=wp.transform((1.5, -1.5, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
            linear_compliance=1e-3,
            angular_compliance=1e-4,
        )
        # Right wheel revolute joint (velocity control)
        builder.add_joint_revolute(
            parent=chassis,
            child=right_wheel,
            parent_xform=wp.transform((1.5, 1.5, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
            linear_compliance=1e-3,
            angular_compliance=1e-4,
        )
        # Back wheel revolute joint (force control - not actively driven)
        builder.add_joint_revolute(
            parent=chassis,
            child=back_wheel,
            parent_xform=wp.transform((-2.5, 0.0, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=wp.sim.JOINT_MODE_FORCE,
            linear_compliance=1e-3,
            angular_compliance=1e-4,
        )

        # --- Add Static Obstacles and Ground ---

        # Add a static box obstacle (body=-1 means it's fixed to the world)
        builder.add_shape_box(
            body=-1,
            pos=wp.vec3(5.0, 0.0, 0.0),
            hx=2.5,
            hy=3.0,
            hz=0.3,
            mu=FRICTION,
            restitution=RESTITUTION,
        )
        builder.add_shape_box(
            body=-1,
            pos=wp.vec3(5.0, 0.0, 0.0),
            hx=1.5,
            hy=2.5,
            hz=0.5,
            mu=FRICTION,
            restitution=RESTITUTION,
        )
        # builder.add_shape_box(
        #     body=-1,
        #     pos=wp.vec3(5.0, 1.0, 0.0),
        #     hx=1.0,
        #     hy=1.0,
        #     hz=0.75,
        #     mu=FRICTION,
        #     restitution=RESTITUTION,
        # )

        builder.set_ground_plane(
            ke=10.0,
            kd=10.0,
            kf=0.0,
            mu=FRICTION,
            restitution=RESTITUTION,
        )

        model = builder.finalize()
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_simple_example(cfg: DictConfig):
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
    helhest_simple_example()
