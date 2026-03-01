
from importlib.resources import files
from typing import override
import pathlib

import hydra
import newton
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig
from pendulum_articulation_definition import build_pendulum_model, PENDULUM_HEIGHT
#from axion.core.control_utils import JointMode

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

CONFIG_PATH = CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")

class Simulator(InteractiveSimulator):
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

    @override
    def control_policy(self, state: newton.State):
        # TODO: add later if needed 
        #wp.copy(self.control.joint_f, wp.array([0.0, 800.0], dtype=wp.float32))
        pass

    @override
    def _render(self, segment_num: int):
        """Renders the current state to the appropriate viewers, including world XYZ axes."""
        sim_time = segment_num * self.steps_per_segment * self.clock.dt
        self.viewer.begin_frame(sim_time)
        self.viewer.log_state(self.current_state)
        self.viewer.log_contacts(self.contacts, self.current_state)
        
        # Draw world axes at origin
        axis_length = 1.0  # Length of each axis
        origin = wp.vec3(0.0, 0.0, 0.0)
        
        # Define axis endpoints
        x_end = wp.vec3(axis_length, 0.0, 0.0)  # X axis (red)
        y_end = wp.vec3(0.0, axis_length, 0.0)  # Y axis (green)
        z_end = wp.vec3(0.0, 0.0, axis_length)  # Z axis (blue)
        
        # Create arrays for line starts and ends
        device = wp.get_device()
        starts = wp.array([origin, origin, origin], dtype=wp.vec3, device=device)
        ends = wp.array([x_end, y_end, z_end], dtype=wp.vec3, device=device)
        
        # Colors: red for X, green for Y, blue for Z
        colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0),  # Red for X
             wp.vec3(0.0, 1.0, 0.0),  # Green for Y
             wp.vec3(0.0, 0.0, 1.0)], # Blue for Z
            dtype=wp.vec3,
            device=device
        )
        
        # Draw the axes
        self.viewer.log_lines("world_axes", starts, ends, colors, width=0.08)
        
        # Draw reference frame at the first pendulum link anchor point
        anchor_x = 0.0
        anchor_y = 0.0
        anchor_z = PENDULUM_HEIGHT  # Position from parent_xform
        anchor_axis_length = 0.5  # Slightly shorter than world axes
        
        # Define axis endpoints (absolute positions)
        anchor_point = wp.vec3(anchor_x, anchor_y, anchor_z)
        anchor_x_end = wp.vec3(anchor_x + anchor_axis_length, anchor_y, anchor_z)
        anchor_y_end = wp.vec3(anchor_x, anchor_y + anchor_axis_length, anchor_z)
        anchor_z_end = wp.vec3(anchor_x, anchor_y, anchor_z + anchor_axis_length)
        
        # Create arrays for anchor frame lines
        anchor_starts = wp.array(
            [anchor_point, anchor_point, anchor_point],
            dtype=wp.vec3,
            device=device
        )
        anchor_ends = wp.array(
            [anchor_x_end, anchor_y_end, anchor_z_end],
            dtype=wp.vec3,
            device=device
        )
        
        # Same colors as world axes
        anchor_colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0),  # Red for X
             wp.vec3(0.0, 1.0, 0.0),  # Green for Y
             wp.vec3(0.0, 0.0, 1.0)], # Blue for Z
            dtype=wp.vec3,
            device=device
        )
        
        # Draw the anchor reference frame
        self.viewer.log_lines("anchor_frame", anchor_starts, anchor_ends, anchor_colors, width=0.08)
        
        self.viewer.end_frame()

    def build_model(self,) -> newton.Model:
        """
        Use the same pendulum articulation as axionEnv.py
        """
        return build_pendulum_model(num_worlds=1, device="cuda:0") # is it necessary to pass device?


@hydra.main(config_path=str(CONFIG_PATH), config_name="neuralPendulum", version_base=None)
def basic_pendulum_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    basic_pendulum_example()