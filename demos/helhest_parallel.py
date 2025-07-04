import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import warp.sim.render
from axion.nsn_engine import NSNEngine
from tqdm import tqdm
from warp.sim import Mesh

# wp.config.mode = "debug"
# wp.config.verify_cuda = True
# wp.config.verify_fp = True

RENDER = False
DEBUG = True
USD_FILE = "helhest.usd"

FRICTION = 1.0
RESTITUTION = 0.8


def ball_world_model(gravity: bool = True) -> wp.sim.Model:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

    left_wheel = builder.add_body(
        origin=wp.transform((1.5, -1.5, 1.2), wp.quat_identity()), name="ball1"
    )
    builder.add_shape_sphere(
        body=left_wheel,
        radius=1.0,
        density=10.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        collision_group=1,
    )

    right_wheel = builder.add_body(
        origin=wp.transform((1.5, 1.5, 1.2), wp.quat_identity()), name="ball2"
    )

    builder.add_shape_sphere(
        body=right_wheel,
        radius=1.0,
        density=10.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        collision_group=1,
    )

    back_wheel = builder.add_body(
        origin=wp.transform((-2.5, 0.0, 1.2), wp.quat_identity()), name="ball3"
    )

    builder.add_shape_sphere(
        body=back_wheel,
        radius=1.0,
        density=10.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        collision_group=1,
    )

    obstacle1 = builder.add_body(
        origin=wp.transform((0.0, 0.0, 1.2), wp.quat_identity()), name="box1"
    )

    builder.add_shape_box(
        body=obstacle1,
        hx=1.5,
        hy=0.5,
        hz=0.5,
        density=10.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        collision_group=1,
    )

    builder.add_joint_revolute(
        parent=obstacle1,
        child=left_wheel,
        parent_xform=wp.transform((1.5, -1.5, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        linear_compliance=0.0,
        angular_compliance=0.0,
        mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
    )
    builder.add_joint_revolute(
        parent=obstacle1,
        child=right_wheel,
        parent_xform=wp.transform((1.5, 1.5, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        linear_compliance=0.0,
        angular_compliance=0.0,
        mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
    )
    builder.add_joint_revolute(
        parent=obstacle1,
        child=back_wheel,
        parent_xform=wp.transform((-2.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        linear_compliance=0.0,
        angular_compliance=0.0,
        mode=wp.sim.JOINT_MODE_FORCE,
    )

    obstacle1 = builder.add_body(
        origin=wp.transform((0.0, 0.0, 1.2), wp.quat_identity()), name="box1"
    )

    builder.add_shape_box(
        body=-1,
        pos=wp.vec3(4.5, 0.0, 0.0),
        hx=1.5,
        hy=2.5,
        hz=0.3,
        density=10.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        collision_group=1,
    )

    builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=FRICTION, restitution=RESTITUTION)
    model = builder.finalize()
    return model


class BallBounceSim:
    def __init__(self):

        # Simulation and rendering parameters
        self.fps = 10
        self.num_frames = 30
        self.sim_substeps = 15
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        self.sim_steps = int(self.sim_duration // self.sim_dt)

        self.model = ball_world_model(gravity=True)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = wp.sim.XPBDIntegrator(
        #     enable_restitution=True, rigid_contact_relaxation=0.0
        # )

        self.integrator = NSNEngine(self.model)
        if RENDER:
            self.renderer = wp.sim.render.SimRenderer(
                self.model, USD_FILE, scaling=100.0
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.control.joint_act = wp.array(
            [
                3.0,  # left wheel
                3.0,  # right wheel
                0.0,  # back wheel
            ],
            dtype=wp.float32,
        )
        self.model.joint_target_ke = wp.full(
            (self.model.joint_count,), value=500.0, dtype=wp.float32
        )

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.multistep()
            self.step_graph = capture.graph

    def multistep(self):
        for _ in range(self.sim_substeps):
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(
                self.model,
                self.state_0,
                self.state_1,
                self.sim_dt,
                control=self.control,
            )
            wp.copy(dest=self.state_0.body_q, src=self.state_1.body_q)
            wp.copy(dest=self.state_0.body_qd, src=self.state_1.body_qd)

    def step(self):
        wp.sim.collide(self.model, self.state_0)
        self.integrator.simulate(
            self.model, self.state_0, self.state_1, self.sim_dt, control=self.control
        )

        wp.copy(dest=self.state_0.body_q, src=self.state_1.body_q)
        wp.copy(dest=self.state_0.body_qd, src=self.state_1.body_qd)

    def simulate(self):
        frame_interval = 1.0 / self.fps
        last_rendered_time = 0.0

        for i in tqdm(range(self.sim_steps), desc="Simulating", disable=DEBUG):
            with wp.ScopedTimer("step", active=DEBUG):
                if self.use_cuda_graph:
                    wp.capture_launch(self.step_graph)
                else:
                    self.step()
                wp.synchronize()
            if RENDER:
                with wp.ScopedTimer("render", active=DEBUG):
                    t = self.time[i]
                    if t >= last_rendered_time:  # render only if enough time has passed
                        self.renderer.begin_frame(t)
                        self.renderer.render(self.state_0)
                        self.renderer.end_frame()
                        last_rendered_time += (
                            frame_interval  # update to next frame time
                        )
                    wp.synchronize()

        if RENDER:
            self.renderer.save()

    def simulate_multistep(self):
        t = 0.0
        frame_interval = 1.0 / self.fps
        last_rendered_time = 0.0
        for i in tqdm(range(self.num_frames), desc="Simulating", disable=DEBUG):
            with wp.ScopedTimer("step", active=DEBUG):
                if self.use_cuda_graph:
                    wp.capture_launch(self.step_graph)
                else:
                    self.multistep()
            t += self.frame_dt
            if RENDER:
                with wp.ScopedTimer("render", active=DEBUG):
                    wp.synchronize()
                    if t >= last_rendered_time:
                        self.renderer.begin_frame(t)
                        self.renderer.render(self.state_0)
                        self.renderer.end_frame()
                        last_rendered_time += (
                            frame_interval  # update to next frame time
                        )
        if RENDER:
            self.renderer.save()


def ball_bounce_simulation():
    model0 = BallBounceSim()
    model1 = BallBounceSim()

    stream0 = wp.Stream()
    stream1 = wp.Stream()

    with wp.ScopedStream(stream0):
        model0.simulate_multistep()
    with wp.ScopedStream(stream1):
        model1.simulate_multistep()


if __name__ == "__main__":
    ball_bounce_simulation()
