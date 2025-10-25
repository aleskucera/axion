from importlib.resources import files
from typing import override

import hydra
import newton
import numpy as np
import openmesh
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import ProfilingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

CONFIG_PATH = files("axion").joinpath("examples").joinpath("conf")
ASSETS_DIR = files("axion").joinpath("examples").joinpath("assets")


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

    @override
    def control_policy(self, current_state: newton.State):
        wp.copy(
            self.control.joint_target, wp.array(6 * [0.0] + [10.0, 10.0, 0.0], dtype=wp.float32)
        )

    def build_model(self) -> newton.Model:
        """
        Implements the abstract method to define the physics objects in the scene.

        This method constructs the three-wheeled vehicle, obstacles, and ground plane.
        """
        FRICTION = 1.0
        RESTITUTION = 0.0

        builder = newton.ModelBuilder()
        builder.add_articulation(key="helhest")

        # --- Build the Vehicle ---
        wheel_m = openmesh.read_trimesh(f"{ASSETS_DIR}/helhest/wheel2.obj")
        mesh_points = np.array(wheel_m.points())
        mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_render = newton.Mesh(mesh_points, mesh_indices)

        wheel_m_col = openmesh.read_trimesh(f"{ASSETS_DIR}/helhest/wheel_collision.obj")
        mesh_points = np.array(wheel_m_col.points())
        mesh_indices = np.array(wheel_m_col.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_collision = newton.Mesh(mesh_points, mesh_indices)

        # Create main body (chassis)
        chassis = builder.add_body(
            xform=wp.transform((-2.0, 0.0, 2.6), wp.quat_identity()), key="chassis"
        )
        builder.add_shape_box(
            body=chassis,
            hx=0.75,
            hy=0.25,
            hz=0.25,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=1200.0, mu=FRICTION, restitution=RESTITUTION
            ),
        )

        # Left Wheel
        left_wheel = builder.add_body(
            xform=wp.transform((-1.25, -0.75, 2.6), wp.quat_identity()), key="left_wheel"
        )
        builder.add_shape_mesh(
            body=left_wheel,
            mesh=wheel_mesh_render,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                has_shape_collision=False,
            ),
        )
        builder.add_shape_mesh(
            body=left_wheel,
            mesh=wheel_mesh_collision,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=500.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                is_visible=False,
            ),
        )

        # Right Wheel
        right_wheel = builder.add_body(
            xform=wp.transform((-1.25, 0.75, 2.6), wp.quat_identity()), key="right_wheel"
        )
        builder.add_shape_mesh(
            body=right_wheel,
            mesh=wheel_mesh_render,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                has_shape_collision=False,
            ),
        )

        builder.add_shape_mesh(
            body=right_wheel,
            mesh=wheel_mesh_collision,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=500.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                is_visible=False,
            ),
        )

        # Back Wheel
        back_wheel = builder.add_body(
            xform=wp.transform((-3.25, 0.0, 2.6), wp.quat_identity()), key="back_wheel"
        )
        builder.add_shape_mesh(
            body=back_wheel,
            mesh=wheel_mesh_render,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                has_shape_collision=False,
            ),
        )
        builder.add_shape_mesh(
            body=back_wheel,
            mesh=wheel_mesh_collision,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=500.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                is_visible=False,
            ),
        )

        # --- Define Joints ---

        builder.add_joint_free(parent=-1, child=chassis)

        # Left wheel revolute joint (velocity control)
        builder.add_joint_revolute(
            parent=chassis,
            child=left_wheel,
            parent_xform=wp.transform((0.75, -0.75, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=newton.JointMode.TARGET_VELOCITY,
        )
        # Right wheel revolute joint (velocity control)
        builder.add_joint_revolute(
            parent=chassis,
            child=right_wheel,
            parent_xform=wp.transform((0.75, 0.75, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=newton.JointMode.TARGET_VELOCITY,
        )
        # Back wheel revolute joint (not actively driven)
        builder.add_joint_revolute(
            parent=chassis,
            child=back_wheel,
            parent_xform=wp.transform((-1.5, 0.0, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=newton.JointMode.NONE,
        )

        # Set joint control gains
        builder.joint_target_ke[-3] = 50.0
        builder.joint_target_ke[-2] = 50.0
        builder.joint_target_ke[-1] = 0.0
        builder.joint_armature[-3] = 0.1
        builder.joint_armature[-2] = 0.1
        builder.joint_armature[-1] = 0.1
        builder.joint_target_kd[-3] = 5.0
        builder.joint_target_kd[-2] = 5.0
        builder.joint_target_kd[-1] = 5.0

        # --- Add Static Obstacles and Ground ---

        # Add a static box obstacle (body=-1 means it's fixed to the world)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((2.5, 0.0, 0.0), wp.quat_identity()),
            hx=1.75,
            hy=1.5,
            hz=0.15,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((2.5, 0.0, 0.0), wp.quat_identity()),
            hx=0.75,
            hy=1.75,
            hz=0.25,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        # add ground plane
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=10.0, kd=10.0, kf=0.0, mu=FRICTION, restitution=RESTITUTION
            )
        )

        # Finalize and return the model
        model = builder.finalize()
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_example(cfg: DictConfig):
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
    helhest_example()
