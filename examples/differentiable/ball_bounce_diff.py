import os

import newton
import numpy as np
import warp as wp
import warp.optim
from axion import DifferentiableSimulator
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SemiImplicitEngineConfig
from axion import SimulationConfig
from newton import Model
from newton import ModelBuilder

os.environ["PYOPENGL_PLATFORM"] = "glx"


def bourke_color_map(v_min, v_max, v):
    c = wp.vec3(1.0, 1.0, 1.0)
    v = np.clip(v, v_min, v_max)
    dv = v_max - v_min

    if v < (v_min + 0.25 * dv):
        c[0] = 0.0
        c[1] = 4.0 * (v - v_min) / dv
    elif v < (v_min + 0.5 * dv):
        c[0] = 0.0
        c[2] = 1.0 + 4.0 * (v_min + 0.25 * dv - v) / dv
    elif v < (v_min + 0.75 * dv):
        c[0] = 4.0 * (v - v_min - 0.5 * dv) / dv
        c[2] = 0.0
    else:
        c[1] = 1.0 + 4.0 * (v_min + 0.75 * dv - v) / dv
        c[2] = 0.0

    return c


@wp.kernel
def loss_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_pos: wp.vec3,
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid > 0:
        return

    pos = wp.transform_get_translation(body_q[0])
    delta = pos - target_pos
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def update_kernel(
    qd_grad: wp.array(dtype=wp.spatial_vector),
    alpha: float,
    qd: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    if tid > 0:
        return

    # gradient descent step
    qd[0] = qd[0] - qd_grad[0] * alpha


class BallBounceOptimizer(DifferentiableSimulator):
    def __init__(self):
        sim_config = SimulationConfig(duration_seconds=0.6, target_timestep_seconds=1.0 / 480.0)
        render_config = RenderingConfig()
        exec_config = ExecutionConfig()

        engine_config = SemiImplicitEngineConfig()
        logging_config = LoggingConfig()

        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

        self.target_pos = wp.vec3(0.0, -2.0, 1.5)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.learning_rate = 0.06

        self.frame = 0
        self.frame_dt = 1.0 / 60.0

        # Optimization Parameters
        # Initial velocity guessing (w, v) -> v=(0, 5, -5)
        self.init_vel = wp.spatial_vector(0.0, 5.0, -4.0, 0.0, 0.0, 0.0)
        self.track_body(body_idx=0, name="ball", color=(0.0, 1.0, 0.0))

        self.capture()

    def build_model(self) -> Model:
        shape_config = newton.ModelBuilder.ShapeConfig(ke=1e5, kf=0.0, kd=1e2, mu=0.2)
        builder = ModelBuilder()

        # Initialize the ball
        builder.add_body(
            xform=wp.transform(wp.vec3(0.0, -0.5, 1.0), wp.quat_identity()),
            mass=1.0,
        )
        builder.add_shape_sphere(body=0, radius=0.2, cfg=shape_config)
        # builder.body_qd[0] = [0.0, 5.0, -5.0, 0.0, 0.0, 0.0]

        # Initialize the environment
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()),
            hx=1.0,
            hy=0.30,
            hz=1.0,
            cfg=shape_config,
        )
        builder.add_ground_plane(cfg=shape_config)
        return builder.finalize(requires_grad=True)

    def compute_loss(self) -> wp.array:
        wp.launch(
            kernel=loss_kernel,
            dim=1,
            inputs=[
                self.states[-1].body_q,
                self.target_pos,
            ],
            outputs=[
                self.loss,
            ],
            device=self.solver.model.device,
        )

    def update(self):
        wp.launch(
            kernel=update_kernel,
            dim=1,
            inputs=[
                self.states[0].body_qd.grad,
                self.learning_rate,
            ],
            outputs=[
                self.states[0].body_qd,
            ],
        )

    def render(self, train_iter):
        # Only render every 10 iterations
        if self.frame > 0 and train_iter % 10 != 0:
            return

        # 2. Update the tracked color dynamically based on loss
        loss_val = self.loss.numpy()[0]
        color = bourke_color_map(0.0, 7.0, loss_val)
        # Update the internal config for the track_body system
        self._tracked_bodies[0]["color"] = tuple(color)

        # 3. Define callback for extra visuals (Target & Loss Text)
        def draw_extras(viewer, step_idx, state):
            viewer.log_scalar("/loss", loss_val)

            # Draw target box
            viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                wp.array([wp.transform(self.target_pos, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3),
            )

        # 4. Call render_episode with looping options
        print(f"Rendering iteration {train_iter}...")
        self.render_episode(
            iteration=train_iter,
            callback=draw_extras,
            loop=True,  # Enable looping
            loops_count=1,  # Replay 3 times
            playback_speed=0.3,  # 50% speed (Slow Motion)
        )

        self.frame += 1

    def train(self, iterations=20):
        # Initialize velocity in the first state
        # We set it once, and then optimization updates it
        wp.copy(self.states[0].body_qd, wp.array([self.init_vel], dtype=wp.spatial_vector))
        self.states[0].body_qd.requires_grad = True

        for i in range(iterations):
            self.diff_step()

            self.render(i)

            print(f"Train iter: {i} Loss: {self.loss.numpy()[0]:.4f}")

            self.update()

            self.tape.zero()
            self.loss.zero_()


def main():
    sim = BallBounceOptimizer()
    sim.train(iterations=60)


if __name__ == "__main__":
    main()
