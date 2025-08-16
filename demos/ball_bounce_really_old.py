import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import warp.sim.render
from axion import AxionEngine
from tqdm import tqdm
from warp.sim import Mesh

# wp.config.mode = "debug"
# wp.config.verify_cuda = True
# wp.config.verify_fp = True

RENDER = True
DEBUG = True
USD_FILE = "ball_bounce.usd"
WHEEL_PATH = "data/helhest/wheel.obj"

FRICTION = 0.8
RESTITUTION = 0.8


def ball_world_model(gravity: bool = True) -> wp.sim.Model:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

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

    box_2_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 4.0)
    box2 = builder.add_body(
        origin=wp.transform((0.0, 1.6, 9.0), box_2_rot),
        name="box2",
    )

    builder.add_shape_box(
        body=box2,
        hx=0.8,
        hy=0.8,
        hz=0.8,
        density=10.0,
        ke=2000.0,
        kd=1000.0,
        kf=200.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
    )

    builder.add_joint_revolute(
        parent=box1,
        child=box2,
        parent_xform=wp.transform((0.0, 0.8, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, -0.8, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        linear_compliance=0.0,
        angular_compliance=0.0,
        mode=wp.sim.JOINT_MODE_FORCE,
    )

    builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=FRICTION, restitution=RESTITUTION)
    model = builder.finalize()
    return model


@wp.kernel
def compute_joint_act_kernel(
    joint_qd: wp.array(dtype=wp.float32),
    joint_target_qd: wp.array(dtype=wp.float32),
    kp: wp.array(dtype=wp.float32),
    ki: wp.array(dtype=wp.float32),
    kd: wp.array(dtype=wp.float32),
    joint_qd_sum: wp.array(dtype=wp.float32),
    joint_qd_prev: wp.array(dtype=wp.float32),
    dt: float,
    joint_act: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid < joint_qd.shape[0]:
        error = joint_target_qd[tid] - joint_qd[tid]
        joint_qd_sum[tid] += error * dt
        joint_qd_derivative = (joint_qd[tid] - joint_qd_prev[tid]) / dt
        joint_act[tid] = (
            kp[tid] * error
            + ki[tid] * joint_qd_sum[tid]
            + kd[tid] * joint_qd_derivative
        )
        joint_qd_prev[tid] = joint_qd[tid]


class BallBounceSim:
    def __init__(self):

        # Simulation and rendering parameters
        self.fps = 30
        self.num_frames = 180
        self.sim_substeps = 7
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        self.sim_steps = int(self.sim_duration // self.sim_dt)

        self.model = ball_world_model(gravity=False)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = wp.sim.XPBDIntegrator(
        #     enable_restitution=True, rigid_contact_relaxation=0.0
        # )
        self.integrator = AxionEngine(self.model)
        self.renderer = wp.sim.render.SimRenderer(self.model, USD_FILE, scaling=100.0)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.control.joint_act = wp.full(
            (self.model.joint_count,), value=0.0, dtype=wp.float32
        )
        self.kp = wp.full((self.model.joint_count,), value=500.0, dtype=wp.float32)
        self.ki = wp.full((self.model.joint_count,), value=0.1, dtype=wp.float32)
        self.kd = wp.full((self.model.joint_count,), value=0.2, dtype=wp.float32)
        self.target_joint_qd = wp.full(
            (self.model.joint_count,), value=0.5, dtype=wp.float32
        )
        self.joint_qd_sum = wp.zeros((self.model.joint_count,), dtype=wp.float32)
        self.joint_qd_prev = wp.zeros((self.model.joint_count,), dtype=wp.float32)

        self.use_cuda_graph = wp.get_device().is_cuda and False
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.step()
            self.step_graph = capture.graph

    def _compute_control(self):
        wp.launch(
            kernel=compute_joint_act_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.state_0.joint_qd,
                self.target_joint_qd,
                self.kp,
                self.ki,
                self.kd,
                self.joint_qd_sum,
                self.joint_qd_prev,
                self.frame_dt,
            ],
            outputs=[self.control.joint_act],
        )

    def step(self):
        self._compute_control()
        for i in range(self.sim_substeps):
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

    def simulate(self):
        frame_interval = 1.0 / self.fps
        last_rendered_time = 0.0

        for i in tqdm(range(self.sim_steps), desc="Simulating", disable=DEBUG):
            with wp.ScopedTimer("step", active=DEBUG):
                if self.use_cuda_graph:
                    wp.capture_launch(self.step_graph)
                else:
                    self.step()
            if RENDER:
                with wp.ScopedTimer("render", active=DEBUG):
                    wp.synchronize()
                    t = self.time[i]
                    if t >= last_rendered_time:  # render only if enough time has passed
                        self.renderer.begin_frame(t)
                        self.renderer.render(self.state_0)
                        self.renderer.end_frame()
                        last_rendered_time += (
                            frame_interval  # update to next frame time
                        )

        if RENDER:
            self.renderer.save()


def ball_bounce_simulation():
    model = BallBounceSim()
    model.simulate()


if __name__ == "__main__":
    ball_bounce_simulation()
