import axion
import newton.examples
import warp as wp
import openmesh
import numpy as np

from ._assets import ASSETS_DIR


class HelhestExample:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 2.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.add_articulation(key="helhest")
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )

        FRICTION = 1.0
        RESTITUTION = 0.0

        self.act = (0.0, 0.0)

        # --- Build the Vehicle ---
        wheel_m = openmesh.read_trimesh(f"{ASSETS_DIR}/helhest/wheel2.obj")
        mesh_points = np.array(wheel_m.points())
        mesh_indices = np.array(wheel_m.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_render = newton.Mesh(mesh_points, mesh_indices)

        wheel_m_col = openmesh.read_trimesh(f"{ASSETS_DIR}/helhest/wheel_collision.obj")
        mesh_points = np.array(wheel_m_col.points())
        mesh_indices = np.array(wheel_m_col.face_vertex_indices(), dtype=np.int32).flatten()
        wheel_mesh_collision = newton.Mesh(mesh_points, mesh_indices)

        # Create main body (chassis)
        chassis = builder.add_body(
            xform=wp.transform((0.0, 0.0, 2.6), wp.quat_identity()), key="chassis"
        )
        builder.add_shape_box(
            body=chassis,
            hx=0.75,
            hy=0.25,
            hz=0.25,
            cfg=newton.ModelBuilder.ShapeConfig(density=10.0, mu=FRICTION, restitution=RESTITUTION),
        )

        # Left Wheel
        left_wheel = builder.add_body(
            xform=wp.transform((0.75, -0.75, 2.6), wp.quat_identity()), key="left_wheel"
        )
        builder.add_shape_mesh(
            body=left_wheel,
            mesh=wheel_mesh_render,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=20.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                has_shape_collision=False,
            ),
        )
        builder.add_shape_mesh(
            body=left_wheel,
            mesh=wheel_mesh_collision,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=20.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                is_visible=False,
            ),
        )

        # Right Wheel
        right_wheel = builder.add_body(
            xform=wp.transform((0.75, 0.75, 2.6), wp.quat_identity()), key="right_wheel"
        )
        builder.add_shape_mesh(
            body=right_wheel,
            mesh=wheel_mesh_render,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=20.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                has_shape_collision=False,
            ),
        )

        builder.add_shape_mesh(
            body=right_wheel,
            mesh=wheel_mesh_collision,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=20.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                is_visible=False,
            ),
        )

        # Back Wheel
        back_wheel = builder.add_body(
            xform=wp.transform((-1.5, 0.0, 2.6), wp.quat_identity()), key="back_wheel"
        )
        builder.add_shape_mesh(
            body=back_wheel,
            mesh=wheel_mesh_render,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=20.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                has_shape_collision=False,
            ),
        )
        builder.add_shape_mesh(
            body=back_wheel,
            mesh=wheel_mesh_collision,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=20.0,
                mu=FRICTION,
                restitution=RESTITUTION,
                thickness=0.0,
                is_visible=False,
            ),
        )

        # --- Define Joints ---

        builder.add_joint_free(parent=-1, child=chassis)

        # Left wheel revolute joint (velocity control)
        builder.add_joint_revolute(
            parent=chassis,
            child=left_wheel,
            parent_xform=wp.transform((0.75, -0.75, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=newton.JointMode.TARGET_VELOCITY,
        )
        # Right wheel revolute joint (velocity control)
        builder.add_joint_revolute(
            parent=chassis,
            child=right_wheel,
            parent_xform=wp.transform((0.75, 0.75, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=newton.JointMode.TARGET_VELOCITY,
        )
        # Back wheel revolute joint (force control - not actively driven)
        builder.add_joint_revolute(
            parent=chassis,
            child=back_wheel,
            parent_xform=wp.transform((-1.5, 0.0, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            mode=newton.JointMode.NONE,
        )

        # --- Add Static Obstacles and Ground ---

        # Add a static box obstacle (body=-1 means it's fixed to the world)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((2.5, 0.0, 0.0), wp.quat_identity()),
            hx=1.75,
            hy=1.5,
            hz=0.15,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((2.5, 0.0, 0.0), wp.quat_identity()),
            hx=0.75,
            hy=1.75,
            hz=0.25,
            cfg=newton.ModelBuilder.ShapeConfig(
                mu=FRICTION,
                restitution=RESTITUTION,
            ),
        )

        # add ground plane
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=10.0, kd=10.0, kf=0.0, mu=FRICTION, restitution=RESTITUTION
            )
        )

        builder.joint_target_ke[-3] = 500.0
        builder.joint_target_ke[-2] = 500.0
        builder.joint_target_ke[-1] = 0.0
        builder.joint_armature[-3] = 0.1
        builder.joint_armature[-2] = 0.1
        builder.joint_armature[-1] = 0.1
        builder.joint_target_kd[-3] = 5.0
        builder.joint_target_kd[-2] = 5.0
        builder.joint_target_kd[-1] = 5.0

        # finalize model
        self.model = builder.finalize()

        # self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        # self.solver = newton.solvers.SolverFeatherstone(self.model)
        # self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.solver = axion.AxionEngine(self.model)
        # self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=50)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            left_control = (
                1.0
                if self.viewer.is_key_down("o")
                else (-1.0 if self.viewer.is_key_down("l") else 0.0)
            )
            right_control = (
                1.0
                if self.viewer.is_key_down("i")
                else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            )

            wp.copy(
                self.control.joint_target,
                wp.array(
                    6 * [0.0] + [10 * left_control, 10 * right_control, 0.0],
                    dtype=wp.float32,
                ),
            )

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = HelhestExample(viewer)

    newton.examples.run(example, args)
