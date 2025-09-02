import argparse

import warp as wp
from axion import EngineConfig
from base_simulator import BaseSimulator
from base_simulator import ExecConfig
from base_simulator import ProfileConfig
from base_simulator import RenderConfig
from base_simulator import SimConfig


class BallBounceSimulator(BaseSimulator):
    def __init__(
        self,
        sim_config: SimConfig,
        render_config: RenderConfig,
        exec_config: ExecConfig,
        profile_config: ProfileConfig,
        engine_config: EngineConfig,
    ):
        super().__init__(sim_config, render_config, exec_config, profile_config, engine_config)

    def build_model(self) -> wp.sim.Model:
        FRICTION = 0.8
        RESTITUTION = 0.5

        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        ball1 = builder.add_body(
            origin=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), name="ball1"
        )
        builder.add_shape_sphere(
            body=ball1,
            radius=1.0,
            density=10.0,
            ke=2000.0,
            kd=10.0,
            kf=200.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
        )

        ball2 = builder.add_body(
            origin=wp.transform((0.3, 0.0, 4.5), wp.quat_identity()), name="ball2"
        )

        builder.add_shape_sphere(
            body=ball2,
            radius=1.0,
            density=10.0,
            ke=2000.0,
            kd=10.0,
            kf=200.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
        )

        ball3 = builder.add_body(
            origin=wp.transform((-0.6, 0.0, 6.5), wp.quat_identity()), name="ball3"
        )

        builder.add_shape_sphere(
            body=ball3,
            radius=0.8,
            density=10.0,
            ke=2000.0,
            kd=10.0,
            kf=200.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
        )

        ball4 = builder.add_body(
            origin=wp.transform((-0.6, 0.0, 10.5), wp.quat_identity()), name="ball4"
        )

        builder.add_shape_sphere(
            body=ball4,
            radius=0.5,
            density=10.0,
            ke=2000.0,
            kd=10.0,
            kf=200.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
        )

        box1 = builder.add_body(
            origin=wp.transform((0.0, 0.0, 9.0), wp.quat_identity()), name="box1"
        )

        builder.add_shape_box(
            body=box1,
            hx=0.8,
            hy=0.8,
            hz=0.8,
            density=10.0,
            ke=2000.0,
            kd=10.0,
            kf=200.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
        )

        builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=FRICTION, restitution=RESTITUTION)
        model = builder.finalize()
        return model


def main():
    parser = argparse.ArgumentParser(description="Run the Ball Bounce physics simulation.")

    # SimConfig Arguments
    parser.add_argument(
        "--duration", type=float, default=4.0, help="Total simulation time in seconds."
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
        default="collision_primitives.usd",
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

    sim_config = SimConfig(
        sim_duration=args.duration,
        target_sim_dt=args.dt,
    )

    render_config = RenderConfig(
        enable=not args.headless,
        usd_file=args.outfile,
        fps=args.fps,
    )

    exec_config = ExecConfig(
        use_cuda_graph=not args.no_graph,
    )

    profile_config = ProfileConfig(
        debug=args.debug,
    )

    engine_config = EngineConfig(
        newton_iters=args.newton_iters,
        linear_iters=args.linear_iters,
        linesearch_steps=args.linesearch_steps,
    )

    simulator = BallBounceSimulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        profile_config=profile_config,
        engine_config=engine_config,
    )

    simulator.simulate()


if __name__ == "__main__":
    main()
