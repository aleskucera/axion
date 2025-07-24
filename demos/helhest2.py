import math
import os

import numpy as np
import warp as wp
import warp.sim.render
from axion.nsn_engine import NSNEngine
from tqdm import tqdm

# wp.config.mode = "debug"
# wp.config.verify_cuda = True

###################################
### MAIN SIMULATION CONFIGURATION ###
###################################
RENDER = True
USD_FILE = "helhest2.usd"

FRICTION = 1.0
RESTITUTION = 0.8

#######################################
### PROFILING CONFIGURATION         ###
#######################################
# ADDED: Ported the detailed profiling configuration from helhest.py
# Master switch for all profiling. If False, timers are inactive and have no overhead.
DEBUG = True

# --- ScopedTimer Options ---
# synchronize=True: Waits for all GPU work to finish before stopping the timer.
# This gives you the total wall-clock time (CPU + GPU).
# If False, it only measures the time to *launch* the work on the CPU.
PROFILE_SYNC = True

# use_nvtx=True: Emits NVTX ranges for visualization in NVIDIA Nsight Systems.
# Helps correlate CPU launch times with actual GPU execution on a timeline.
PROFILE_NVTX = False

# cuda_filter=wp.TIMING_ALL: Enables detailed reporting of individual CUDA activities
# (kernels, memory copies, etc.). Prints a detailed timeline and summary.
PROFILE_CUDA_TIMELINE = False

# --- Logic to apply the settings ---
if PROFILE_CUDA_TIMELINE:
    # Use TIMING_ALL to capture kernels, memory copies, and memsets.
    cuda_activity_filter = wp.TIMING_ALL
else:
    cuda_activity_filter = 0  # 0 means disabled


def compute_env_offsets(
    num_envs, env_offset=(5.0, 0.0, 5.0), up_axis="Z"
):  # IMPROVED: Changed up_axis to Z
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0
    env_offsets -= correction
    return env_offsets


def build_single_env(gravity=True) -> wp.sim.ModelBuilder:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

    z_offset = 0.2
    # Add bodies, shapes, and joints for ONE environment
    left_wheel = builder.add_body(
        origin=wp.transform((1.5, -1.5, 1.2 + z_offset), wp.quat_identity()),
        name="ball1",
    )
    builder.add_shape_sphere(
        body=left_wheel, radius=1.0, density=10.0, mu=FRICTION, restitution=RESTITUTION
    )

    right_wheel = builder.add_body(
        origin=wp.transform((1.5, 1.5, 1.2 + z_offset), wp.quat_identity()),
        name="ball2",
    )
    builder.add_shape_sphere(
        body=right_wheel, radius=1.0, density=10.0, mu=FRICTION, restitution=RESTITUTION
    )

    back_wheel = builder.add_body(
        origin=wp.transform((-2.5, 0.0, 1.2 + z_offset), wp.quat_identity()),
        name="ball3",
    )
    builder.add_shape_sphere(
        body=back_wheel, radius=1.0, density=10.0, mu=FRICTION, restitution=RESTITUTION
    )

    obstacle1 = builder.add_body(
        origin=wp.transform((0.0, 0.0, 1.2 + z_offset), wp.quat_identity()), name="box1"
    )
    builder.add_shape_box(
        body=obstacle1,
        hx=1.5,
        hy=0.5,
        hz=0.5,
        density=10.0,
        mu=FRICTION,
        restitution=RESTITUTION,
    )

    # Add joints for wheels
    builder.add_joint_revolute(
        parent=obstacle1,
        child=left_wheel,
        parent_xform=wp.transform((1.5, -1.5, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
    )
    builder.add_joint_revolute(
        parent=obstacle1,
        child=right_wheel,
        parent_xform=wp.transform((1.5, 1.5, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        mode=wp.sim.JOINT_MODE_TARGET_VELOCITY,
    )
    builder.add_joint_revolute(
        parent=obstacle1,
        child=back_wheel,
        parent_xform=wp.transform((-2.5, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
        mode=wp.sim.JOINT_MODE_FORCE,
    )
    return builder


class BallBounceSim:
    def __init__(self, num_envs=8):
        # Simulation and rendering parameters
        self.fps = 10
        self.num_frames = 30
        self.sim_substeps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        # REMOVED: self.sim_steps as the main loop logic is now per-frame

        self.num_envs = num_envs
        main_builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

        # Create single environment template
        env_builder = build_single_env()

        # Compute environment offsets for parallel environments
        offsets = compute_env_offsets(
            num_envs, env_offset=(10.0, 10.0, 0.0), up_axis="Z"
        )

        # Add multiple environment instances
        for i in range(num_envs):
            main_builder.add_builder(
                env_builder, xform=wp.transform(offsets[i], wp.quat_identity())
            )

        main_builder.set_ground_plane(
            ke=10, kd=10, kf=0.0, mu=FRICTION, restitution=RESTITUTION
        )

        self.model = main_builder.finalize()

        # Setup joint control for all environments
        self.control = self.model.control()
        self.control.joint_act = wp.array([3.0, 3.0, 0.0] * num_envs, dtype=wp.float32)

        # Integrator and renderer setup
        self.integrator = NSNEngine(self.model)
        self.renderer = wp.sim.render.SimRenderer(
            self.model, USD_FILE, scaling=50.0
        )  # IMPROVED: Adjusted scaling

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

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

    def simulate(self):
        t = 0.0
        for _ in tqdm(range(self.num_frames), desc="Simulating Frames", disable=DEBUG):
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
            if RENDER:
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

        if RENDER:
            self.renderer.save()


def ball_bounce_simulation():
    model = BallBounceSim(
        num_envs=12
    )  # Change number of environments as needed (e.g., 9 for a 3x3 grid)
    model.simulate()


if __name__ == "__main__":
    ball_bounce_simulation()
