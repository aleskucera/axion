from importlib.resources import files
from typing import override

import hydra
import newton
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig
from axion.core.control_utils import JointMode

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

CONFIG_PATH = files("axion").joinpath("examples").joinpath("conf")

PENDULUM_HEIGHT = 5.0

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

    @override
    def control_policy(self, state: newton.State):
        wp.copy(self.control.joint_f, wp.array([0.0, 800.0], dtype=wp.float32))

    @override
    def _render(self, segment_num: int):
        """Renders the current state to the appropriate viewers, including world XYZ axes."""
        sim_time = segment_num * self.steps_per_segment * self.effective_timestep
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

    def build_model(self) -> newton.Model:

        chain_width = 1.5
        shape_ke = 1.0e4
        shape_kd = 1.0e3
        shape_kf = 1.0e4

        hx = chain_width*0.5

        link_0 = self.builder.add_link(armature=0.1)
        link_config = newton.ModelBuilder.ShapeConfig(density=500.0, ke = shape_ke, kd = shape_kd, kf = shape_kf)
        capsule_shape_transform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi/2))
        
        self.builder.add_shape_capsule(link_0,
                                        xform= capsule_shape_transform,
                                        radius=0.1, 
                                        half_height=chain_width*0.5,
                                        cfg = link_config)

        link_1 = self.builder.add_link(armature=0.1)
        self.builder.add_shape_capsule(link_1,
                                    xform = capsule_shape_transform,
                                    radius=0.1, 
                                    half_height=chain_width*0.5,
                                    cfg = link_config)

        #rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi * 0.5)
        
        j0 = self.builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, PENDULUM_HEIGHT), q= wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q= wp.quat_identity()),
            target_ke=1000.0,
            target_kd=50.0,
            custom_attributes={
                "joint_target_ki": [0.5],
                "joint_dof_mode": [JointMode.NONE],
            },
        )
        j1 = self.builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=500.0,
            target_kd=5.0,
            custom_attributes={
                "joint_target_ki": [0.5],
                "joint_dof_mode": [JointMode.NONE],
            },
            armature=0.1,
        )

        # Create articulation from joints
        self.builder.add_articulation([j0, j1], key="pendulum")

        self.builder.add_ground_plane()

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
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