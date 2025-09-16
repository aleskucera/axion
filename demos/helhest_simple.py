import argparse

import warp as wp
from axion import EngineConfig
from base_simulator import AbstractSimulator
from base_simulator import ExecutionConfig
from base_simulator import ProfilingConfig
from base_simulator import RenderingConfig
from base_simulator import SimulationConfig


class HelhestSimulator(AbstractSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        profile_config: ProfilingConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(sim_config, render_config, exec_config, profile_config, engine_config)

        self.control.joint_act = wp.array(
            [
                3.0,  # left wheel target velocity
                3.0,  # right wheel target velocity
                0.0,  # back wheel (force control, so velocity is 0)
            ],
            dtype=wp.float32,
        )

        # Set the stiffness (motor strength) for the velocity-controlled joints.
        self.model.joint_target_ke = wp.array(
            [
                500.0,  # left wheel
                500.0,  # right wheel
                0.0,  # back wheel
            ],
            dtype=wp.float32,
        )

    def build_model(self) -> wp.sim.Model:
        FRICTION = 0.9
        RESTITUTION = 0.0

        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # --- Build the Vehicle ---

        # Create main body (chassis)
        chassis = builder.add_body(
            origin=wp.transform((0.0, 0.0, 1.2), wp.quat_identity()), name="chassis"
        )
        builder.add_shape_box(
            body=chassis,
            hx=1.5,
            hy=0.5,
            hz=0.5,
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
        )

        # Left Wheel
        left_wheel = builder.add_body(
            origin=wp.transform((1.5, -1.5, 1.2), wp.quat_identity()), name="left_wheel"
        )
        builder.add_shape_sphere(
            body=left_wheel,
            radius=1.0,
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.1,  # Visual flair
        )

        # Right Wheel
        right_wheel = builder.add_body(
            origin=wp.transform((1.5, 1.5, 1.2), wp.quat_identity()), name="right_wheel"
        )
        builder.add_shape_sphere(
            body=right_wheel,
            radius=1.0,
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
        )

        # Back Wheel
        back_wheel = builder.add_body(
            origin=wp.transform((-2.5, 0.0, 1.2), wp.quat_identity()), name="back_wheel"
        )
        builder.add_shape_sphere(
            body=back_wheel,
            radius=1.0,
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
        )

        # --- Define Joints ---

        # Left wheel revolute joint (velocity control)
        builder.add_joint_revolute(
            parent=chassis,
            child=left_wheel,
            parent_xform=wp.transform((1.5, -1.5, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
        )
        # Right wheel revolute joint (velocity control)
        builder.add_joint_revolute(
            parent=chassis,
            child=right_wheel,
            parent_xform=wp.transform((1.5, 1.5, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
        )
        # Back wheel revolute joint (force control - not actively driven)
        builder.add_joint_revolute(
            parent=chassis,
            child=back_wheel,
            parent_xform=wp.transform((-2.5, 0.0, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=wp.sim.JOINT_MODE_FORCE,
        )

        # --- Add Static Obstacles and Ground ---

        # Add a static box obstacle (body=-1 means it's fixed to the world)
        builder.add_shape_box(
            body=-1,
            pos=wp.vec3(4.5, 0.0, 0.0),
            hx=1.5,
            hy=2.5,
            hz=0.3,
            mu=FRICTION,
            restitution=RESTITUTION,
        )

        builder.set_ground_plane(
            ke=10.0,
            kd=10.0,
            kf=0.0,
            mu=FRICTION,
            restitution=RESTITUTION,
        )

        model = builder.finalize()
        return model


def main():
    parser = argparse.ArgumentParser(description="Run the Helhest physics simulation.")

    # SimConfig Arguments
    parser.add_argument(
        "--duration", type=float, default=3.0, help="Total simulation time in seconds."
    )
    parser.add_argument(
        "--dt", type=float, default=5e-3, help="Target physics timestep (dt) in seconds."
    )

    # RenderConfig Arguments
    parser.add_argument(
        "--headless", action="store_true", help="Disable rendering and run in headless mode."
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="helhest_simple.usd",
        help="Output file path for the USD render.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the rendered output."
    )

    # ExecConfig Arguments
    parser.add_argument(
        "--no-graph", action="store_true", help="Disable the use of CUDA graphs for execution."
    )

    # ProfileConfig Arguments
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (disables optimizations)."
    )

    # EngineConfig Arguments
    parser.add_argument(
        "--newton-iters", type=int, default=6, help="Number of Newton iterations for the solver."
    )
    parser.add_argument(
        "--linear-iters", type=int, default=4, help="Number of linear solver iterations."
    )
    parser.add_argument(
        "--linesearch-steps", type=int, default=2, help="Number of linesearch steps in the solver."
    )

    args = parser.parse_args()

    sim_config = SimulationConfig(
        sim_duration=args.duration,
        target_sim_dt=args.dt,
    )

    render_config = RenderingConfig(
        enable=not args.headless,
        usd_file=args.outfile,
        fps=args.fps,
    )

    exec_config = ExecutionConfig(
        use_cuda_graph=not args.no_graph,
    )

    profile_config = ProfilingConfig(
        debug=args.debug,
    )

    engine_config = EngineConfig(
        newton_iters=args.newton_iters,
        linear_iters=args.linear_iters,
        linesearch_steps=args.linesearch_steps,
    )

    simulator = HelhestSimulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        profile_config=profile_config,
        engine_config=engine_config,
    )

    simulator.run()


if __name__ == "__main__":
    main()
