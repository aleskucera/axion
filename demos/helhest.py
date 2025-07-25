import numpy as np
import warp as wp
import warp.sim.render
from axion.nsn_engine import NSNEngine
from tqdm import tqdm

# wp.config.mode = "debug"
# wp.config.verify_cuda = True
# wp.config.verify_fp = True

###################################
### MAIN SIMULATION CONFIGURATION ###
###################################
RENDER = True
USD_FILE = "helhest.usd"

FRICTION = 1.0
RESTITUTION = 0.9

#######################################
### PROFILING CONFIGURATION (MARKO) ###
#######################################
# Master switch for all profiling. If False, timers are inactive and have no overhead.
DEBUG = True

# --- ScopedTimer Options ---

# synchronize=True: Waits for all GPU work to finish before stopping the timer.
# This gives you the total wall-clock time (CPU + GPU).
# If False, it only measures the time to *launch* the work on the CPU, which is often misleadingly short.
PROFILE_SYNC = True

# use_nvtx=True: Emits NVTX ranges for visualization in NVIDIA Nsight Systems.
# Helps correlate CPU launch times with actual GPU execution on a timeline.
# Requires `pip install nvtx`.
PROFILE_NVTX = False

# cuda_filter=wp.TIMING_ALL: Enables detailed reporting of individual CUDA activities
# (kernels, memory copies, etc.). Prints a detailed timeline and summary.
# This is the most insightful option for text-based GPU profiling.
PROFILE_CUDA_TIMELINE = False

# --- Logic to apply the settings ---
if PROFILE_CUDA_TIMELINE:
    # Use TIMING_ALL to capture kernels, memory copies, and memsets.
    # You could also be more specific, e.g., wp.TIMING_KERNEL | wp.TIMING_MEMCPY
    cuda_activity_filter = wp.TIMING_ALL
else:
    cuda_activity_filter = 0  # 0 means disabled


def ball_world_model(gravity: bool = True) -> wp.sim.Model:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

    left_wheel = builder.add_body(
        origin=wp.transform((1.5, -1.5, 2.2), wp.quat_identity()),
        name="ball1",
    )
    builder.add_shape_sphere(
        body=left_wheel,
        radius=1.0,
        density=20.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        collision_group=1,
    )

    right_wheel = builder.add_body(
        origin=wp.transform((1.5, 1.5, 2.2), wp.quat_identity()),
        name="ball2",
    )

    builder.add_shape_sphere(
        body=right_wheel,
        radius=1.0,
        density=20.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        collision_group=1,
    )

    back_wheel = builder.add_body(
        origin=wp.transform((-2.5, 0.0, 2.2), wp.quat_identity()),
        name="ball3",
    )

    builder.add_shape_sphere(
        body=back_wheel,
        radius=1.0,
        density=20.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        collision_group=1,
    )

    robot_base = builder.add_body(
        origin=wp.transform((0.0, 0.0, 2.2), wp.quat_identity()), name="box1"
    )

    builder.add_shape_box(
        body=robot_base,
        hx=1.5,
        hy=0.5,
        hz=0.5,
        density=20.0,
        mu=FRICTION,
        restitution=RESTITUTION,
        thickness=0.0,
        collision_group=1,
    )

    builder.add_joint_revolute(
        parent=robot_base,
        child=left_wheel,
        parent_xform=wp.transform((1.5, -1.5, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        linear_compliance=0.01,
        angular_compliance=0.001,
        mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
    )
    builder.add_joint_revolute(
        parent=robot_base,
        child=right_wheel,
        parent_xform=wp.transform((1.5, 1.5, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        linear_compliance=0.01,
        angular_compliance=0.001,
        mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
    )
    builder.add_joint_revolute(
        parent=robot_base,
        child=back_wheel,
        parent_xform=wp.transform((-2.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        linear_compliance=0.01,
        angular_compliance=0.001,
        mode=wp.sim.JOINT_MODE_FORCE,
    )

    # builder.add_shape_box(
    #     body=-1,
    #     pos=wp.vec3(4.5, 0.0, 0.0),
    #     hx=1.5,
    #     hy=2.5,
    #     hz=0.3,
    #     density=100.0,
    #     mu=FRICTION,
    #     restitution=RESTITUTION,
    #     thickness=0.0,
    #     collision_group=1,
    # )

    builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=FRICTION, restitution=RESTITUTION)
    model = builder.finalize()
    return model


class HelhestSim:
    def __init__(self):

        # Simulation and rendering parameters
        self.fps = 10
        self.num_frames = 30
        self.sim_substeps = 20
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        self.sim_steps = int(self.sim_duration // self.sim_dt)

        self.model = ball_world_model(gravity=True)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        self.integrator = NSNEngine(self.model)
        self.renderer = wp.sim.render.SimRenderer(self.model, USD_FILE, scaling=100.0)

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
        self.model.joint_target_ke = wp.array(
            [
                300.0,  # left wheel
                300.0,  # right wheel
                0.0,  # back wheel
            ],
            dtype=wp.float32,
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
                t = self.time[i]
                if t >= last_rendered_time:  # render only if enough time has passed
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
                    last_rendered_time += frame_interval  # update to next frame time

        if RENDER:
            self.renderer.save()

    def simulate_multistep(self):
        t = 0.0
        frame_interval = 1.0 / self.fps
        last_rendered_time = 0.0
        for _ in tqdm(range(self.num_frames), desc="Simulating", disable=DEBUG):
            with wp.ScopedTimer(
                "step",
                active=DEBUG,
                synchronize=PROFILE_SYNC,
                use_nvtx=PROFILE_NVTX,
                color="blue",  # You can also customize colors for NVTX
                cuda_filter=cuda_activity_filter,
            ):
                if self.use_cuda_graph:
                    wp.capture_launch(self.step_graph)
                else:
                    self.multistep()

            t += self.frame_dt
            if RENDER:
                if t >= last_rendered_time:
                    with wp.ScopedTimer(
                        "render",
                        active=DEBUG,
                        synchronize=PROFILE_SYNC,
                        use_nvtx=PROFILE_NVTX,
                        color="green",
                        cuda_filter=cuda_activity_filter,
                    ):
                        self.renderer.begin_frame(t)
                        self.renderer.render(self.state_0)
                        self.renderer.end_frame()
                    last_rendered_time += frame_interval  # update to next frame time
        if RENDER:
            self.renderer.save()


def ball_bounce_simulation():
    model = HelhestSim()
    model.simulate_multistep()


if __name__ == "__main__":
    ball_bounce_simulation()
