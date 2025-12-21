import os
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
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

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

        # self.mujoco_solver = newton.solvers.SolverMuJoCo(self.model, njmax=40)
        # self.joint_target = wp.array(6 * [0.0] + [0.5, 0.5, 0.0], dtype=wp.float32)

        robot_joint_target = np.concatenate(
            [np.zeros(6), np.array([400.0, 400.0, 0.0], dtype=wp.float32)]
        )

        joint_target = np.tile(robot_joint_target, self.simulation_config.num_worlds)
        self.joint_target = wp.from_numpy(joint_target, dtype=wp.float32)

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
        # wp.copy(self.control.joint_f, self.joint_target)
        pass

    def build_model(self) -> newton.Model:
        """
        Implements the abstract method to define the physics objects in the scene.

        This method constructs the three-wheeled vehicle, obstacles, and ground plane.
        """
        FRICTION = 0.0
        RESTITUTION = 0.0
        WHEEL_DENSITY = 300
        CHASSIS_DENSITY = 800
        KE = 60000.0
        KD = 30000.0
        KF = 500.0

        ball_x = 0.5
        ball_y = 0.0
        ball_z = 2.0

        wheel_m = openmesh.read_trimesh(f"{ASSETS_DIR}/helhest/wheel2.obj")
        mesh_points = np.array(wheel_m.points())
        mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_render = newton.Mesh(mesh_points, mesh_indices)

        # Left Wheel
        left_wheel = self.builder.add_body(
            xform=wp.transform((ball_x, ball_y, ball_z), wp.quat_identity()),
            key="left_wheel",
        )
        self.builder.add_shape_sphere(
            body=left_wheel,
            xform=wp.transform(
                (0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2)
            ),
            radius=0.45,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=WHEEL_DENSITY,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                contact_margin=0.1,
                is_visible=True,
                ke=KE,
                kd=KD,
                kf=KF,
            ),
        )

        # Right Wheel
        left_wheel = self.builder.add_body(
            xform=wp.transform((ball_x + 4.05, ball_y, ball_z), wp.quat_identity()),
            key="right_wheel",
        )
        self.builder.add_shape_sphere(
            body=left_wheel,
            xform=wp.transform(
                (0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2)
            ),
            radius=0.45,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=WHEEL_DENSITY,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                contact_margin=0.1,
                is_visible=True,
                ke=KE,
                kd=KD,
                kf=KF,
            ),
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
                thickness=0.0,
                contact_margin=0.5,
            ),
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds,
            gravity=-9.81,
            rigid_contact_margin=0.0,
        )


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
