from importlib.resources import files
from typing import override
import pathlib

import hydra
import numpy as np
import newton
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig
from axion.core.model_builder import AxionModelBuilder
from axion import JointMode

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

#CONFIG_PATH = files("axion").joinpath("examples").joinpath("conf")
CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")

PENDULUM_HEIGHT = 5.0


# ---------------------------------------------------------------------------
# Helper: generalized → maximal coordinate conversion
# ---------------------------------------------------------------------------

def generalized_to_maximal(
    model: newton.Model,
    state: newton.State,
    q0: float,
    q1: float,
    qd0: float = 0.0,
    qd1: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert generalized pendulum coordinates to maximal coordinates
    and write them into *state*.

    Uses ``newton.eval_fk`` (forward kinematics) to map joint-space
    quantities to body-space transforms and spatial velocities.

    Args:
        model:  The Newton ``Model`` that describes the double pendulum.
        state:  A ``newton.State`` whose ``joint_q``, ``joint_qd``,
                ``body_q`` and ``body_qd`` will be **overwritten**.
        q0:     Joint-0 angle  (revolute, radians).
        q1:     Joint-1 angle  (revolute, radians).
        qd0:    Joint-0 angular velocity  (rad/s, default 0).
        qd1:    Joint-1 angular velocity  (rad/s, default 0).

    Returns:
        body_q  – ``np.ndarray`` of shape ``(num_bodies, 7)``
                  ``[x, y, z, qx, qy, qz, qw]`` per body.
        body_qd – ``np.ndarray`` of shape ``(num_bodies, 6)``
                  ``[vx, vy, vz, wx, wy, wz]`` per body.

    Example::

        body_q, body_qd = generalized_to_maximal(
            model, state, q0=0.5, q1=-0.3, qd0=1.0, qd1=-2.0,
        )
    """
    device = state.joint_q.device

    # Write generalized coordinates into the state
    state.joint_q.assign(
        wp.array([q0, q1], dtype=wp.float32, device=device)
    )
    state.joint_qd.assign(
        wp.array([qd0, qd1], dtype=wp.float32, device=device)
    )

    # Forward kinematics: joint_q/joint_qd  →  body_q/body_qd
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    body_q_np = state.body_q.numpy().reshape(-1, 7)
    body_qd_np = state.body_qd.numpy().reshape(-1, 6)
    return body_q_np, body_qd_np

class Simulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
        initial_state: tuple[float, float, float, float] | None = None,
    ):
        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
            logging_config,
        )

        # --- Apply custom initial conditions (positions AND velocities) ---
        if initial_state is not None:
            q0, q1, qd0, qd1 = initial_state
            generalized_to_maximal(
                self.model, self.current_state,
                q0=q0, q1=q1, qd0=qd0, qd1=qd1,
            )

    @override
    def control_policy(self, state: newton.State):
        wp.copy(self.control.joint_f, wp.array([0.0, 800.0], dtype=wp.float32))

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

    def build_model(
        self,
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

        return builder.finalize_replicated(num_worlds=1)

@hydra.main(config_path=str(CONFIG_PATH), config_name="pendulum", version_base=None)
def basic_pendulum_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    # Custom initial conditions: (q0, q1, qd0, qd1)
    # Set to None to start from the default rest position.
    INITIAL_STATE = (-4.684e-1, 2.077, -2.048, 4.629)  # e.g. (0.5, -0.3, 1.0, -2.0)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
        logging_config=logging_config,
        initial_state=INITIAL_STATE,
    )

    simulator.run()


if __name__ == "__main__":
    basic_pendulum_example()