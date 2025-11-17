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
from axion import JointMode
from axion import LoggingConfig
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
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
            logging_config,
        )

        self.mujoco_solver = newton.solvers.SolverMuJoCo(self.model, njmax=40)
        self.joint_target = wp.array(6 * [0.0] + [0.5, 0.5, 0.0], dtype=wp.float32)

    @override
    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        # self.mujoco_solver.step(current_state, next_state, self.model.control(), contacts, dt)
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state: newton.State):
        # For axion
        wp.copy(self.control.joint_target, self.joint_target)

    def build_model(self) -> newton.Model:
        """
        Implements the abstract method to define the physics objects in the scene.

        This method constructs the three-wheeled vehicle, obstacles, and ground plane.
        """
        FRICTION = 0.8
        RESTITUTION = 0.0
        WHEEL_DENSITY = 300
        CHASSIS_DENSITY = 800

        wheel_m = openmesh.read_trimesh(f"{ASSETS_DIR}/helhest/wheel2.obj")
        mesh_points = np.array(wheel_m.points())
        mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_render = newton.Mesh(mesh_points, mesh_indices)

        self.builder.add_articulation(key="helhest_simple")

        # --- Build the Vehicle ---
        # Create main body (chassis)
        chassis = self.builder.add_body(
            xform=wp.transform((-2.0, 0.0, 1.0), wp.quat_identity()), key="chassis"
        )
        self.builder.add_shape_box(
            body=chassis,
            hx=0.3,
            hy=0.45,
            hz=0.2,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=CHASSIS_DENSITY,
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        # Left Wheel
        left_wheel = self.builder.add_body(
            xform=wp.transform((-2.0, -0.75, 1.0), wp.quat_identity()),
            key="left_wheel",
        )
        self.builder.add_shape_mesh(
            body=left_wheel,
            mesh=wheel_mesh_render,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=False,
            ),
        )
        self.builder.add_shape_capsule(
            body=left_wheel,
            xform=wp.transform(
                (0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2)
            ),
            radius=0.45,
            half_height=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=WHEEL_DENSITY,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                is_visible=False,
            ),
        )

        # Right Wheel
        right_wheel = self.builder.add_body(
            xform=wp.transform((-2.0, 0.75, 1.0), wp.quat_identity()),
            key="right_wheel",
        )
        self.builder.add_shape_mesh(
            body=right_wheel,
            mesh=wheel_mesh_render,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=False,
            ),
        )
        self.builder.add_shape_capsule(
            body=right_wheel,
            xform=wp.transform(
                (0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2)
            ),
            radius=0.45,
            half_height=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=WHEEL_DENSITY,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                is_visible=False,
            ),
        )

        # Back Wheel
        back_wheel = self.builder.add_body(
            xform=wp.transform((-3.25, 0.0, 1.0), wp.quat_identity()),
            key="back_wheel",
        )
        self.builder.add_shape_mesh(
            body=back_wheel,
            mesh=wheel_mesh_render,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0,
                has_shape_collision=False,
            ),
        )
        self.builder.add_shape_capsule(
            body=back_wheel,
            xform=wp.transform(
                (0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2)
            ),
            radius=0.45,
            half_height=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=WHEEL_DENSITY,
                mu=0.5,
                restitution=RESTITUTION,
                thickness=0.0,
                is_visible=False,
            ),
        )

        # --- Define Joints ---

        self.builder.add_joint_free(parent=-1, child=chassis)

        # Left wheel revolute joint (velocity control)
        self.builder.add_joint_revolute(
            parent=chassis,
            child=left_wheel,
            parent_xform=wp.transform((0.0, -0.75, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            target_ke=400.0,
            target_kd=40.5,
            custom_attributes={
                "joint_target_ki": [0.5],
                "joint_dof_mode": [JointMode.NONE],
            },
        )
        # Right wheel revolute joint (velocity control)
        self.builder.add_joint_revolute(
            parent=chassis,
            child=right_wheel,
            parent_xform=wp.transform((0.0, 0.75, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            target_ke=400.0,
            target_kd=40.5,
            custom_attributes={
                "joint_target_ki": [0.5],
                "joint_dof_mode": [JointMode.NONE],
            },
        )
        # Back wheel revolute joint (not actively driven)
        self.builder.add_joint_revolute(
            parent=chassis,
            child=back_wheel,
            parent_xform=wp.transform((-1.5, 0.0, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
        )

        # --- Add Static Obstacles and Ground ---

        # Add a static box obstacle (body=-1 means it's fixed to the world)
        self.builder.add_shape_box(
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
        self.builder.add_shape_box(
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
        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=10.0, kd=10.0, kf=0.0, mu=FRICTION, restitution=RESTITUTION
            )
        )

        # Finalize and return the model
        model = self.builder.finalize()
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_simple_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    helhest_simple_example()
