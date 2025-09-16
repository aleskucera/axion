import argparse

import numpy as np
import openmesh
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
        """
        Implements the abstract method to define the physics objects in the scene.

        This method constructs the three-wheeled vehicle, obstacles, and ground plane.
        """
        FRICTION = 0.9
        RESTITUTION = 0.0

        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # --- Build the Vehicle ---
        wheel_m = openmesh.read_trimesh("data/helhest/wheel2.obj")
        mesh_points = np.array(wheel_m.points())
        mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_render = wp.sim.Mesh(mesh_points, mesh_indices)

        wheel_m_col = openmesh.read_trimesh("data/helhest/wheel_collision.obj")
        mesh_points = np.array(wheel_m_col.points())
        mesh_indices = np.array(wheel_m_col.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_collision = wp.sim.Mesh(mesh_points, mesh_indices)

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
        builder.add_shape_mesh(
            body=left_wheel,
            mesh=wheel_mesh_render,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            has_ground_collision=False,
            has_shape_collision=False,
        )
        builder.add_shape_mesh(
            body=left_wheel,
            mesh=wheel_mesh_collision,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            is_visible=False,
        )

        # Right Wheel
        right_wheel = builder.add_body(
            origin=wp.transform((1.5, 1.5, 1.2), wp.quat_identity()), name="right_wheel"
        )
        builder.add_shape_mesh(
            body=right_wheel,
            mesh=wheel_mesh_render,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            has_ground_collision=False,
            has_shape_collision=False,
        )
        builder.add_shape_mesh(
            body=right_wheel,
            mesh=wheel_mesh_collision,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            is_visible=False,
        )

        # Back Wheel
        back_wheel = builder.add_body(
            origin=wp.transform((-2.5, 0.0, 1.2), wp.quat_identity()), name="back_wheel"
        )
        builder.add_shape_mesh(
            body=back_wheel,
            mesh=wheel_mesh_render,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            has_ground_collision=False,
            has_shape_collision=False,
        )
        builder.add_shape_mesh(
            body=back_wheel,
            mesh=wheel_mesh_collision,
            scale=(2.0, 2.0, 2.0),
            density=10.0,
            mu=FRICTION,
            restitution=RESTITUTION,
            thickness=0.0,
            is_visible=False,
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

        # surface_m = openmesh.read_trimesh("data/surface.obj")
        # mesh_points = np.array(surface_m.points())
        # mesh_indices = np.array(surface_m.face_vertex_indices(), dtype=np.int32).flatten()
        # surface_mesh = wp.sim.Mesh(mesh_points, mesh_indices)
        #
        # builder.add_shape_mesh(
        #     body=-1,
        #     mesh=surface_mesh,
        #     pos=wp.vec3(0.0, 0.0, -0.2),
        #     scale=(5.0, 5.0, 1.0),
        #     density=10.0,
        #     mu=FRICTION,
        #     restitution=RESTITUTION,
        #     thickness=0.0,
        #     has_ground_collision=False,
        # )

        model = builder.finalize()
        model.ground = True
        return model


def main():
    parser = argparse.ArgumentParser(description="Run the Helhest physics simulation.")

    # SimConfig Arguments
    parser.add_argument(
        "--duration", type=float, default=3.0, help="Total simulation time in seconds."
    )
    parser.add_argument(
        "--dt", type=float, default=1e-3, help="Target physics timestep (dt) in seconds."
    )

    # RenderConfig Arguments
    parser.add_argument(
        "--headless", action="store_true", help="Disable rendering and run in headless mode."
    )
    parser.add_argument(
        "--outfile", type=str, default="helhest.usd", help="Output file path for the USD render."
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
        "--newton-iters", type=int, default=8, help="Number of Newton iterations for the solver."
    )
    parser.add_argument(
        "--linear-iters", type=int, default=4, help="Number of linear solver iterations."
    )
    parser.add_argument(
        "--linesearch-steps", type=int, default=0, help="Number of linesearch steps in the solver."
    )

    args = parser.parse_args()

    # 1. Populate configuration dataclasses from parsed arguments
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

    # 2. Instantiate the simulator with the configurations.
    simulator = HelhestSimulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        profile_config=profile_config,
        engine_config=engine_config,
    )

    # 3. Run the simulation.
    simulator.run()


if __name__ == "__main__":
    main()
