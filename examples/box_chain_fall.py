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
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")


class Simulator(AbstractSimulator):
    def __init__(self, sim_config, render_config, exec_config, engine_config, logging_config):
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

    def build_model(self) -> newton.Model:
        # 1. Ground Plane
        self.builder.add_ground_plane(0.0)

        # 2. Chain Parameters
        num_links = 20
        # Box dimensions
        hx, hy, hz = 0.3, 0.1, 0.45
        # Link length = 2 * hx = 0.6
        box_length = 2.0 * hx
        # Spacing: slightly larger than box length to avoid initial overlap if desired, 
        # or exactly box length if joints are at edges.
        # User had link_spacing = 0.65 in previous example (gap of 0.05).
        link_spacing = 0.65

        # Initial Position (High up)
        start_pos = np.array([0.0, 0.0, 5.0])
        
        all_joints = []

        # Head Link
        # Orient along X axis
        q_identity = wp.quat_identity()
        X_head = wp.transform(wp.vec3(start_pos[0], start_pos[1], start_pos[2]), q_identity)

        head = self.builder.add_link(key="head", mass=1.0, xform=X_head)
        self.builder.add_shape_box(
            head,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=newton.ModelBuilder.ShapeConfig(
                is_visible=True,
                has_shape_collision=True, # Enable collision
                density=100.0,
            ),
        )
        # Free joint for the head (so it can move/fall)
        head_joint = self.builder.add_joint(newton.JointType.FREE, -1, head)
        all_joints.append(head_joint)

        prev_link = head
        
        for i in range(1, num_links):
            # Position subsequent links along X axis
            # x_pos = start_x + i * spacing
            curr_pos = start_pos + np.array([i * link_spacing, 0.0, 0.0])
            
            X_curr = wp.transform(wp.vec3(curr_pos[0], curr_pos[1], curr_pos[2]), q_identity)

            curr_link = self.builder.add_link(key=f"link_{i}", mass=1.0, xform=X_curr)
            self.builder.add_shape_box(
                curr_link,
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=newton.ModelBuilder.ShapeConfig(
                    is_visible=True,
                    has_shape_collision=True,
                    density=100.0,
                ),
            )

            # Ball Joint connecting Prev -> Curr
            # Prev is at X. Curr is at X + spacing.
            # Joint should be halfway? Or at edges?
            # If spacing > length, there is a gap.
            # Let's put the joint anchor at the edge of the boxes + half gap.
            # Distance between centers = link_spacing.
            # Anchor for Prev (at Left): +link_spacing/2
            # Anchor for Curr (at Right): -link_spacing/2
            # This centers the joint exactly between the two centroids.
            
            X_p_joint = wp.transform(wp.vec3(link_spacing / 2.0, 0.0, 0.0), wp.quat_identity())
            X_c_joint = wp.transform(wp.vec3(-link_spacing / 2.0, 0.0, 0.0), wp.quat_identity())

            ball_joint = self.builder.add_joint(
                newton.JointType.BALL,
                prev_link,
                curr_link,
                parent_xform=X_p_joint,
                child_xform=X_c_joint,
                # Use standard compliance or the stiff one from before
                custom_attributes={"joint_compliance": 1e-4}, # Slightly compliant for stability
            )
            all_joints.append(ball_joint)

            prev_link = curr_link

        self.builder.add_articulation(all_joints, key="chain_arti")

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def box_chain_fall_example(cfg: DictConfig):
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    
    # We use XPBD or Standard solver. Default "pos" constraint level is good for stiff contacts.
    engine_config.joint_constraint_level = "pos"
    engine_config.contact_constraint_level = "pos"
    
    # Global Compliance
    engine_config.joint_compliance = 1e-5 # Relatively stiff
    engine_config.contact_compliance = 1e-5 # Hard contacts

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = Simulator(sim_config, render_config, exec_config, engine_config, logging_config)
    simulator.run()


if __name__ == "__main__":
    box_chain_fall_example()
