from importlib.resources import files

import hydra
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

from ._assets import ASSETS_DIR

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
        """
        Implements the abstract method to define the physics objects in the scene.

        This method constructs the three-wheeled vehicle, obstacles, and ground plane.
        """
        FRICTION = 1.0
        RESTITUTION = 0.0

        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # --- Build the Vehicle ---
        wheel_m = openmesh.read_trimesh(f"{ASSETS_DIR}/helhest/wheel2.obj")
        mesh_points = np.array(wheel_m.points())
        mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_render = wp.sim.Mesh(mesh_points, mesh_indices)

        wheel_m_col = openmesh.read_trimesh(f"{ASSETS_DIR}/helhest/wheel_collision.obj")
        mesh_points = np.array(wheel_m_col.points())
        mesh_indices = np.array(wheel_m_col.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_collision = wp.sim.Mesh(mesh_points, mesh_indices)

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
        builder.add_shape_mesh(
            body=left_wheel,
            mesh=wheel_mesh_render,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            has_ground_collision=False,
            has_shape_collision=False,
        )
        builder.add_shape_mesh(
            body=left_wheel,
            mesh=wheel_mesh_collision,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            is_visible=False,
        )

        # Right Wheel
        right_wheel = builder.add_body(
            origin=wp.transform((1.5, 1.5, 1.2), wp.quat_identity()), name="right_wheel"
        )
        builder.add_shape_mesh(
            body=right_wheel,
            mesh=wheel_mesh_render,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            has_ground_collision=False,
            has_shape_collision=False,
        )
        builder.add_shape_mesh(
            body=right_wheel,
            mesh=wheel_mesh_collision,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            is_visible=False,
        )

        # Back Wheel
        back_wheel = builder.add_body(
            origin=wp.transform((-2.5, 0.0, 1.2), wp.quat_identity()), name="back_wheel"
        )
        builder.add_shape_mesh(
            body=back_wheel,
            mesh=wheel_mesh_render,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            has_ground_collision=False,
            has_shape_collision=False,
        )
        builder.add_shape_mesh(
            body=back_wheel,
            mesh=wheel_mesh_collision,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            is_visible=False,
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
        model.ground = True
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
