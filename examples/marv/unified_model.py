import os
import pathlib
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion import LoggingConfig
from common import create_marv_model
from omegaconf import DictConfig
# Import your local marv creator

os.environ["PYOPENGL_PLATFORM"] = "glx"
# Point to existing conf directory (assuming examples/conf exists)
CONFIG_PATH = pathlib.Path(__file__).parent.parent.joinpath("conf")


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

        # Control Mapping
        # Marv has:
        # 1 Free Joint (7 DOFs: 3 pos + 4 quat) OR (6 DOFs if handled as spatial vec) -> Axion usually 6 for free
        # 4 Flippers (Position Control)
        # 16 Wheels (Position Control)

        # Axion Joint Layout usually: [Base, Joints...]
        # We need to target indices.
        # Base: 0-5
        # Flippers: 6, 11, 16, 21 (Assuming 5 joints per leg: 1 flipper + 4 wheels)
        # See creation order in common.py:
        #   create_flipper_leg adds [Flipper, Wheel1, Wheel2, Wheel3, Wheel4]

        # Let's define default targets
        # Flippers: 0.0 rad (flat)
        # Wheels: driving forward

        # We construct a full target vector.
        # Total DOFs = 6 (Base) + 4 * 5 = 26.

        self.num_dofs = 26
        
        self.drive_speed = 10.0
        self.wheel_pos = 0.0
        
        # Initial targets (zeros)
        target_np = np.zeros(self.num_dofs, dtype=np.float32)

        # Replicate for num_worlds
        full_target = np.tile(target_np, self.simulation_config.num_worlds)
        self.joint_targets = wp.from_numpy(full_target, dtype=wp.float32, device=self.model.device)

    @override
    def init_state_fn(self, current_state, next_state, contacts, dt):
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    @override
    def control_policy(self, current_state):
        # Update wheel position
        dt = self.effective_timestep
        self.wheel_pos += self.drive_speed * dt
        
        # Build target array on CPU
        target_np = np.zeros(self.num_dofs, dtype=np.float32)
        
        for leg in range(4):
            base_idx = 6 + leg * 5
            # target_np[base_idx] = 0.0 # Flipper Position
            target_np[base_idx + 1 : base_idx + 5] = self.wheel_pos

        # Replicate and upload
        full_target = np.tile(target_np, self.simulation_config.num_worlds)
        wp.copy(self.joint_targets, wp.array(full_target, dtype=wp.float32, device=self.model.device))

        # Apply targets to the control buffer
        wp.copy(self.control.joint_target_pos, self.joint_targets)

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

    simulator = Simulator(
        sim_config,
        render_config,
        exec_config,
        engine_config,
        logging_config,
    )
    simulator.run()


if __name__ == "__main__":
    marv_example()
