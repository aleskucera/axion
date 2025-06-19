import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import warp.optim
import warp.sim.render
from axion.nsn_engine import NSNEngine
from tqdm import tqdm

DEBUG = True
USD_FILE = "ball_bounce.usd"
PLOT_FILE = "ball_bounce_plot.png"


def ball_world_model(gravity: bool = True) -> wp.sim.Model:
    if gravity:
        builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
    else:
        builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

    b = builder.add_body(
        origin=wp.transform((0.0, 0.0, 2.5), wp.quat_identity()), name="ball"
    )
    builder.add_shape_sphere(
        body=b,
        radius=1.0,
        density=10.0,
        ke=2000.0,
        kd=10.0,
        kf=200.0,
        mu=1.0,
        restitution=1.0,
        thickness=0.05,
    )
    builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=1.0, restitution=1.0)
    model = builder.finalize()

    return model


@wp.kernel
def update_trajectory_kernel(
    trajectory: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.transform),
    time_step: wp.int32,
    q_idx: wp.int32,
):
    trajectory[time_step] = wp.transform_get_translation(q[q_idx])


@wp.kernel
def trajectory_loss_kernel(
    trajectory: wp.array(dtype=wp.vec3f),
    target_trajectory: wp.array(dtype=wp.vec3f),
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    diff = trajectory[tid] - target_trajectory[tid]
    distance_loss = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, distance_loss)


class BallBounceSim:
    def __init__(self):

        # Simulation and rendering parameters
        self.fps = 30
        self.num_frames = 120
        self.sim_substeps = 5
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        self.sim_steps = int(self.sim_duration // self.sim_dt)

        self.model = ball_world_model(gravity=True)
        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        self.integrator = NSNEngine()
        self.renderer = wp.sim.render.SimRenderer(self.model, USD_FILE, scaling=100.0)
        self.render = False  # Set to True to enable rendering

        self.state_in = self.model.state()
        self.state_out = self.model.state()

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.step()
            self.graph = capture.graph

    def simulate(self):
        frame_interval = 1.0 / self.fps  # time interval per frame
        last_rendered_time = 0.0  # tracks the time of the last rendered frame

        with wp.ScopedTimer("Simulation", active=DEBUG):
            for i in tqdm(range(self.sim_steps), desc="Simulating", disable=DEBUG):
                with wp.ScopedTimer(f"Step {i + 1}/{self.sim_steps}", active=DEBUG):
                    if self.use_cuda_graph:
                        wp.capture_launch(self.graph)
                    else:
                        self.step()

                if self.render and self.time[i] >= last_rendered_time:
                    self.renderer.begin_frame(self.time[i])
                    self.renderer.render(self.state_in)
                    self.renderer.end_frame()
                    last_rendered_time += frame_interval

        self.renderer.save()

    def step(self):
        wp.sim.collide(self.model, self.state_in)
        self.integrator.simulate(self.model, self.state_in, self.state_out, self.sim_dt)
        self.state_in, self.state_out = self.state_out, self.state_in

    def save_trajectory(self):
        """Save the trajectory as matplotlib png plot."""
        trajectory_data = self.trajectory.numpy()

        x = trajectory_data[:, 0]
        y = trajectory_data[:, 1]
        z = trajectory_data[:, 2]

        # Make 3 subplots for x, y, z
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        axs[0].plot(self.time, x, label="X Position", color="blue")
        axs[0].set_title("X Position Over Time")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("X Position")

        axs[1].plot(self.time, y, label="Y Position", color="green")
        axs[1].set_title("Y Position Over Time")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Y Position")

        axs[2].plot(self.time, z, label="Z Position", color="red")
        axs[2].set_title("Z Position Over Time")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Z Position")

        plt.tight_layout()
        plt.savefig(PLOT_FILE)

        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plot: {e}")


def ball_bounce_simulation():
    sim = BallBounceSim()
    sim.simulate()


if __name__ == "__main__":
    ball_bounce_simulation()
