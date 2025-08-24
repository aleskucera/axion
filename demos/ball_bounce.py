import numpy as np
import warp as wp
import warp.sim.render
from axion import AxionEngine
from axion import EngineConfig
from axion.utils import HDF5Logger
from tqdm import tqdm

# Options
RENDER = True
USD_FILE = "ball_bounce.usd"
DEBUG = True
PROFILE_SYNC = False
PROFILE_NVTX = False
PROFILE_CUDA_TIMELINE = False

FRICTION = 0.8
RESTITUTION = 0.5

# Optional micro-profiling visibility settings
np.set_printoptions(suppress=False, precision=2)
if PROFILE_CUDA_TIMELINE:
    cuda_activity_filter = wp.TIMING_ALL
else:
    cuda_activity_filter = 0


def ball_world_model(gravity: bool = True) -> wp.sim.Model:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

    ball1 = builder.add_body(origin=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()), name="ball1")
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

    ball2 = builder.add_body(origin=wp.transform((0.3, 0.0, 4.5), wp.quat_identity()), name="ball2")

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

    box1 = builder.add_body(origin=wp.transform((0.0, 0.0, 9.0), wp.quat_identity()), name="box1")

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

    # box_2_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 4.0)
    # box2 = builder.add_body(
    #     origin=wp.transform((0.0, 1.6, 9.0), box_2_rot),
    #     name="box2",
    # )
    #
    # builder.add_shape_box(
    #     body=box2,
    #     hx=0.8,
    #     hy=0.8,
    #     hz=0.8,
    #     density=10.0,
    #     ke=2000.0,
    #     kd=1000.0,
    #     kf=200.0,
    #     mu=FRICTION,
    #     restitution=RESTITUTION,
    #     thickness=0.0,
    # )
    #
    # builder.add_joint_revolute(
    #     parent=box1,
    #     child=box2,
    #     parent_xform=wp.transform((0.0, 0.8, 0.0), wp.quat_identity()),
    #     child_xform=wp.transform((0.0, -0.8, 0.0), wp.quat_identity()),
    #     axis=wp.vec3(0.0, 1.0, 0.0),
    #     linear_compliance=0.0,
    #     angular_compliance=0.0,
    #     mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
    # )

    builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=FRICTION, restitution=RESTITUTION)
    model = builder.finalize()
    return model


class BallBounceSim:
    def __init__(self):
        # Time & simulation config
        self.fps = 30
        self.num_frames = 90
        self.sim_substeps = 10
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        self.sim_steps = int(self.sim_duration // self.sim_dt)

        self.model = ball_world_model(gravity=True)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps + 1)
        self._timestep = 0

        self.logger = HDF5Logger("ball_bounce_log.h5") if DEBUG else None

        engine_config = EngineConfig(newton_iters=4, linear_iters=4, linesearch_steps=0)

        self.integrator = AxionEngine(self.model, engine_config, logger=self.logger)
        self.renderer = wp.sim.render.SimRenderer(self.model, USD_FILE, scaling=100.0, fps=self.fps)

        # Alloc states and controls
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.use_cuda_graph = wp.get_device().is_cuda and not DEBUG
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.step()
            self.step_graph = capture.graph

    def log_model(self):
        if self.logger:
            with self.logger.scope("simulation_info"):
                self.logger.log_scalar("fps", self.fps)
                self.logger.log_scalar("num_frames", self.num_frames)
                self.logger.log_scalar("sim_substeps", self.sim_substeps)
                self.logger.log_scalar("frame_dt", self.frame_dt)
                self.logger.log_scalar("sim_dt", self.sim_dt)
                self.logger.log_scalar("sim_duration", self.sim_duration)

            with self.logger.scope("model_info"):
                m = self.model
                self.logger.log_scalar("body_count", m.body_count)
                self.logger.log_scalar("joint_count", m.joint_count)
                self.logger.log_scalar("rigid_contact_max", m.rigid_contact_max)

    def step(self):
        with (
            self.logger.scope(f"timestep_{self._timestep:04d}")
            if self.logger
            else open("/dev/null")
        ) as _:
            wp.sim.collide(self.model, self.state_0)
            # if self.model.rigid_contact_count.numpy()[0] > 0:
            #     print(f"Contact at timestep {self._timestep}")
            self.integrator.simulate(
                self.model,
                self.state_0,
                self.state_1,
                self.sim_dt,
                self.control,
            )
            wp.copy(dest=self.state_0.body_q, src=self.state_1.body_q)
            wp.copy(dest=self.state_0.body_qd, src=self.state_1.body_qd)

    def multistep(self):
        for _ in range(self.sim_substeps):
            with (
                self.logger.scope(f"timestep_{self._timestep:04d}")
                if self.logger
                else open("/dev/null")
            ) as _:
                if self.logger:
                    self.logger.log_attribute("time", self.time[self._timestep])
                    self.logger.log_scalar("time", self.time[self._timestep])
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
                self._timestep += 1

    def simulate(self):
        self._timestep = 0
        frame_interval = 1.0 / self.fps
        last_rendered_time = 0.0

        with self.logger if self.logger else open("/dev/null") as _:
            self.log_model()

            for i in tqdm(range(self.sim_steps), desc="Simulating", disable=DEBUG):
                with wp.ScopedTimer(
                    "step",
                    active=DEBUG,
                    synchronize=PROFILE_SYNC,
                    use_nvtx=PROFILE_NVTX,
                    cuda_filter=cuda_activity_filter,
                ):
                    if self.use_cuda_graph:
                        wp.capture_launch(self.step_graph)
                    else:
                        self.step()

                if RENDER:
                    with wp.ScopedTimer("render", active=DEBUG):
                        wp.synchronize()
                        t = self.time[self._timestep]
                        if t >= last_rendered_time:  # render only if enough time has passed
                            self.renderer.begin_frame(t)
                            self.renderer.render(self.state_0)
                            self.renderer.end_frame()
                            last_rendered_time += frame_interval  # update to next frame time

                self._timestep += 1

            if RENDER:
                self.renderer.save()

    def simulate_multistep(self):
        t = 0.0
        frame_interval = 1.0 / self.fps
        last_rendered_time = 0.0
        with self.logger if self.logger else open("/dev/null") as _:
            self.log_model()
            for _ in tqdm(range(self.num_frames), desc="Simulating", disable=DEBUG):
                with wp.ScopedTimer(
                    "step",
                    active=DEBUG,
                    synchronize=PROFILE_SYNC,
                    use_nvtx=PROFILE_NVTX,
                    cuda_filter=cuda_activity_filter,
                ):
                    if self.use_cuda_graph:
                        wp.capture_launch(self.step_graph)
                    else:
                        self.multistep()

                t += self.frame_dt
                if RENDER and t >= last_rendered_time:
                    with wp.ScopedTimer(
                        "render",
                        active=DEBUG,
                        synchronize=PROFILE_SYNC,
                        use_nvtx=PROFILE_NVTX,
                        cuda_filter=cuda_activity_filter,
                    ):
                        self.renderer.begin_frame(t)
                        self.renderer.render(self.state_0)
                        self.renderer.end_frame()
                    last_rendered_time += frame_interval

            if RENDER:
                self.renderer.save()


def ball_bounce_simulation():
    sim = BallBounceSim()
    sim.simulate()


if __name__ == "__main__":
    ball_bounce_simulation()
