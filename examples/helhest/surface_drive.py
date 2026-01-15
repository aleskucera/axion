import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import openmesh
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

try:
    from examples.helhest.common import create_helhest_model
except ImportError:
    from common import create_helhest_model

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")
ASSETS_DIR = pathlib.Path(__file__).parent.parent.joinpath("assets")


class HelhestSurfaceSimulator(AbstractSimulator):
    def __init__(self, sim_config, render_config, exec_config, engine_config):
        self.left_indices_cpu = []
        self.right_indices_cpu = []
        super().__init__(sim_config, render_config, exec_config, engine_config)

        # Helhest DOFs: 6 (Base) + 3 (Left, Right, Rear)
        self.joint_target = wp.zeros(9, dtype=wp.float32, device=self.model.device)

    @override
    def _run_simulation_segment(self, segment_num: int):
        self._update_input()
        super()._run_simulation_segment(segment_num)

    def _update_input(self):
        """Check keyboard input and update wheel velocities."""
        base_speed = 5.0
        turn_speed = 2.5

        left_v = 0.0
        right_v = 0.0

        # Simple WASD/Arrow style logic
        if hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("i"):  # Forward
                left_v += base_speed
                right_v += base_speed
            if self.viewer.is_key_down("k"):  # Backward
                left_v -= base_speed
                right_v -= base_speed

            # Turn Left/Right
            if self.viewer.is_key_down("j"):  # Left
                left_v -= turn_speed
                right_v += turn_speed
            if self.viewer.is_key_down("l"):  # Right
                left_v += turn_speed
                right_v -= turn_speed

        rear_v = (left_v + right_v) / 2.0

        # Update targets
        targets_cpu = np.zeros(9, dtype=np.float32)
        targets_cpu[6] = left_v
        targets_cpu[7] = right_v
        targets_cpu[8] = rear_v

        wp.copy(
            self.joint_target, wp.array(targets_cpu, dtype=wp.float32, device=self.model.device)
        )

    @override
    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state: newton.State):
        wp.copy(self.control.joint_target, self.joint_target)

    def build_model(self) -> newton.Model:
        """
        Builds the unified Helhest model on a surface.
        """

        # Robot position
        robot_x = -1.5
        robot_y = 0.0
        robot_z = 1.7

        create_helhest_model(
            self.builder, xform=wp.transform((robot_x, robot_y, robot_z), wp.quat_identity())
        )

        # Surface Mesh
        surface_m = openmesh.read_trimesh(str(ASSETS_DIR.joinpath("surface.obj")))
        mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()

        scale = np.array([6.0, 6.0, 4.0])
        mesh_points = np.array(surface_m.points()) * scale + np.array([0.0, 0.0, 0.05])

        surface_mesh = newton.Mesh(mesh_points, mesh_indices)

        self.builder.add_shape_mesh(
            body=-1,
            mesh=surface_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.0, has_shape_collision=True, mu=1.0, contact_margin=0.3
            ),
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def helhest_surface_drive_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = HelhestSurfaceSimulator(sim_config, render_config, exec_config, engine_config)
    simulator.run()


if __name__ == "__main__":
    helhest_surface_drive_example()
