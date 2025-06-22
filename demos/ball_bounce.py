import matplotlib.pyplot as plt
import numpy as np
import warp as wp
import warp.optim
import warp.sim.render
from axion.nsn_engine2 import NSNEngine

wp.config.mode = "debug"
wp.config.verify_cuda = True
# wp.config.verify_fp = True

DEBUG = False
USD_FILE = "ball_bounce.usd"
PLOT_FILE = "ball_bounce_trajectory.png"


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
        mu=1.0,
        restitution=0.5,
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
        mu=1.0,
        restitution=0.5,
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
        mu=1.0,
        restitution=0.5,
        thickness=0.0,
    )

    ball4 = builder.add_body(
        origin=wp.transform((-0.6, 0.0, 10.5), wp.quat_identity()), name="ball3"
    )

    builder.add_shape_sphere(
        body=ball4,
        radius=0.5,
        density=10.0,
        ke=2000.0,
        kd=10.0,
        kf=200.0,
        mu=1.0,
        restitution=0.5,
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
        mu=1.0,
        restitution=0.5,
        thickness=0.0,
    )

    # builder.body_qd[box1] = wp.spatial_vector(0.35, 0.0, 0.0, 0.0, 0.0, 0.0)

    builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=1.0, restitution=0.8)
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
        self.sim_substeps = 10
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
        self.renderer = wp.sim.render.SimRenderer(self.model, USD_FILE, scaling=1.0)

        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        self.trajectory = wp.empty(len(self.time), dtype=wp.vec3, requires_grad=True)

    def forward(self):
        for i in range(self.sim_steps):
            with wp.ScopedTimer("Collision Detection"):
                wp.sim.collide(self.model, self.states[i])
            with wp.ScopedTimer("Simulation Step"):
                self.integrator.simulate(
                    self.model, self.states[i], self.states[i + 1], self.sim_dt
                )
                wp.launch(
                    kernel=update_trajectory_kernel,
                    dim=1,
                    inputs=[self.trajectory, self.states[i].body_q, i, 0],
                )

    def save_usd(self, fps: int = 30):
        frame_interval = 1.0 / fps  # time interval per frame
        last_rendered_time = 0.0  # tracks the time of the last rendered frame

        print("Creating USD render...")
        for t, state in zip(self.time, self.states):
            if t >= last_rendered_time:  # render only if enough time has passed
                self.renderer.begin_frame(t)
                self.renderer.render(state)
                self.renderer.end_frame()
                last_rendered_time += frame_interval  # update to next frame time

        self.renderer.save()

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
    model = BallBounceSim()
    model.forward()
    model.save_usd()
    model.save_trajectory()


if __name__ == "__main__":
    ball_bounce_simulation()
