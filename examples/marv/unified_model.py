import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from common import create_marv_model
from omegaconf import DictConfig
# Import your local marv creator

os.environ["PYOPENGL_PLATFORM"] = "glx"
# Point to existing conf directory (assuming examples/conf exists)
CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


class Simulator(AbstractSimulator):
    def __init__(self, sim_config, render_config, exec_config, engine_config, logging_config):
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        # Control Mapping
        # Marv has:
        # 1 Free Joint (7 DOFs: 3 pos + 4 quat) OR (6 DOFs if handled as spatial vec) -> Axion usually 6 for free
        # 4 Flippers (Position Control)
        # 16 Wheels (Velocity Control)

        # Axion Joint Layout usually: [Base, Joints...]
        # We need to target indices.
        # Base: 0-5
        # Flippers: 6, 11, 16, 21 (Assuming 5 joints per leg: 1 flipper + 4 wheels)
        # See creation order in common.py:
        #   create_flipper_leg adds [Flipper, Wheel1, Wheel2, Wheel3, Wheel4]

        # Let's define default targets
        # Flippers: 0.0 rad (flat)
        # Wheels: 5.0 rad/s (drive forward)

        # We construct a full target vector.
        # Total DOFs = 6 (Base) + 4 * 5 = 26.

        self.num_dofs = 26
        target_np = np.zeros(self.num_dofs, dtype=np.float32)

        # Set Wheel Velocity Targets (Indices relative to start of articulation DOFs)
        # Each leg has 5 joints.
        # Leg 1 (FL): Joint 6 is flipper, 7-10 are wheels.
        # Leg 2 (FR): Joint 11 is flipper, 12-15 are wheels.
        # ...

        drive_speed = 10.0

        for leg in range(4):
            base_idx = 6 + leg * 5
            # target_np[base_idx] = 0.0 # Flipper Position
            target_np[base_idx + 1 : base_idx + 5] = drive_speed  # Wheel Velocities

        # Replicate for num_worlds
        full_target = np.tile(target_np, self.simulation_config.num_worlds)
        self.joint_targets = wp.from_numpy(full_target, dtype=wp.float32)

    @override
    def init_state_fn(self, current_state, next_state, contacts, dt):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state):
        # Apply targets to the control buffer
        wp.copy(self.control.joint_target, self.joint_targets)

    def build_model(self) -> newton.Model:
        # Create Marv at a slight height
        create_marv_model(self.builder, xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))

        # Add Ground
        self.builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                contact_margin=0.05, ke=1.0e4, kd=1.0e3, kf=1.0e3, mu=0.8, restitution=0.0
            )
        )

        # Optional: Add the palette obstacles from XML?
        # self.builder.add_shape_box(...)

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, gravity=-9.81
        )


# Reuse the taros-4 config or create a generic one
@hydra.main(config_path=str(CONFIG_PATH), config_name="taros-4", version_base=None)
def marv_example(cfg: DictConfig):
    sim_config = hydra.utils.instantiate(cfg.simulation)
    render_config = hydra.utils.instantiate(cfg.rendering)
    exec_config = hydra.utils.instantiate(cfg.execution)
    engine_config = hydra.utils.instantiate(cfg.engine)
    logging_config = hydra.utils.instantiate(cfg.logging)

    simulator = Simulator(sim_config, render_config, exec_config, engine_config, logging_config)
    simulator.run()


if __name__ == "__main__":
    marv_example()
