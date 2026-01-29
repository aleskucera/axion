import os
import pathlib
from typing import override

import hydra
import newton
import warp as wp
from axion import InteractiveSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SimulationConfig
from axion.core.control_utils import JointMode
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")


@wp.kernel
def compute_control(
    dt: wp.float32,
    time_seconds: wp.array(dtype=wp.float32),
    joint_target: wp.array(dtype=wp.float32),
):
    wp.atomic_add(time_seconds, 0, dt)
    t = time_seconds[0]
    # if t < 0.1:
    #     return

    joint_target[0] = 1.0 * wp.sin(0.5 * wp.pi * t)
    # wp.printf("Joint target: %f \n", joint_target[0])


class Simulator(InteractiveSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
        )
        self.time = wp.zeros(1, dtype=wp.float32)

    @override
    def control_policy(self, state: newton.State):
        # wp.copy(
        #     self.control.joint_target,
        #     wp.array([target_pos], dtype=wp.float32, device=self.control.joint_target.device),
        # )
        wp.launch(
            compute_control,
            dim=1,
            inputs=[self.effective_timestep, self.time],
            outputs=[self.control.joint_target],
        )

    def build_model(self) -> newton.Model:
        # Standard cylinder is along Z. Rotate 90 deg around X to align with Y.
        q_rod = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5)

        # Disable collision for the rod
        rod_cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
        self.builder.add_shape_cylinder(
            body=-1,
            radius=0.05,
            half_height=50.0,  # Total length 10.0
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=q_rod),
            cfg=rod_cfg,
        )

        # 2. Create the sliding box
        hx = 0.5
        hy = 0.5
        hz = 0.5
        link_0 = self.builder.add_link()
        # self.slider_body is no longer needed for control, but keeping the link reference is fine
        self.builder.add_shape_box(link_0, hx=hx, hy=hy, hz=hz)

        # 3. Add Prismatic Joint along Y-axis
        j0 = self.builder.add_joint_prismatic(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),  # Y axis
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
            child_xform=wp.transform_identity(),
            # Add limits
            limit_lower=-4.0,
            limit_upper=4.0,
        )

        # Create articulation
        self.builder.add_articulation([j0], key="slider")

        self.builder.add_ground_plane()

        model = self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)

        # Configure Control Mode: Position Control
        wp.copy(
            model.joint_dof_mode,
            wp.array([int(JointMode.TARGET_POSITION)], dtype=wp.int32, device=model.device),
        )
        # Stiffness (Kp)
        wp.copy(
            model.joint_target_ke,
            wp.array([1500.0], dtype=wp.float32, device=model.device),
        )
        # Damping (Kd)
        wp.copy(
            model.joint_target_kd,
            wp.array([300.0], dtype=wp.float32, device=model.device),
        )

        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def horizontal_slider_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
    )

    simulator.run()


if __name__ == "__main__":
    horizontal_slider_example()
