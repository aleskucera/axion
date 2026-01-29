import numpy as np
import warp as wp
import warp.optim
from axion import AxionEngineConfig
from axion import DifferentiableSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SemiImplicitEngineConfig
from axion import SimulationConfig
from newton import Model
from newton import ModelBuilder


@wp.kernel
def loss_kernel(
    body_q_traj: wp.array(dtype=wp.transform, ndim=3),
    target_pos: wp.vec3,
    step_count: int,
    loss: wp.array(dtype=float),
):
    last_step = step_count - 1
    # Check bounds just in case
    if last_step < 0:
        return

    pos_xform = body_q_traj[last_step, 0, 0]
    pos = wp.transform_get_translation(pos_xform)
    delta = pos - target_pos
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def map_vel_kernel(v_in: wp.array(dtype=wp.vec3), v_out: wp.array(dtype=wp.spatial_vector, ndim=2)):
    v = v_in[0]
    v_out[0, 0] = wp.spatial_vector(0.0, 0.0, 0.0, v[0], v[1], v[2])


@wp.kernel
def flatten_slice_kernel(
    traj_q: wp.array(dtype=wp.transform, ndim=3),
    step_idx: int,
    flat_q: wp.array(dtype=wp.transform, ndim=1),
    num_bodies: int,
    num_worlds: int,
):
    world_idx, body_idx = wp.tid()
    if world_idx >= num_worlds or body_idx >= num_bodies:
        return
    flat_idx = world_idx * num_bodies + body_idx
    flat_q[flat_idx] = traj_q[step_idx, world_idx, body_idx]


class BallBounceOptimizer(DifferentiableSimulator):
    def __init__(self, engine_type="semi_implicit"):
        sim_config = SimulationConfig(duration_seconds=1.0, target_timestep_seconds=1.0 / 60.0)
        render_config = RenderingConfig(vis_type="null")
        exec_config = ExecutionConfig(use_cuda_graph=False)

        if engine_type == "axion":
            engine_config = AxionEngineConfig(differentiable_simulation=True)
        else:
            engine_config = SemiImplicitEngineConfig()

        super().__init__(sim_config, render_config, exec_config, engine_config)

        self.target_pos = wp.vec3(0.0, -2.0, 1.5)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)

        # Optimization Parameters
        self.init_vel = wp.vec3(0.0, 5.0, -5.0)
        self.param_vel = wp.array([self.init_vel], dtype=wp.vec3, requires_grad=True)

    def build_model(self) -> Model:
        builder = ModelBuilder()
        builder.add_body(xform=wp.transform(wp.vec3(0.0, -0.5, 1.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_sphere(body=0, radius=0.1)
        builder.add_ground_plane()
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()),
            hx=1.0,
            hy=0.25,
            hz=1.0,
        )
        return builder.finalize(requires_grad=True)

    def compute_loss(self) -> wp.array:
        wp.launch(
            kernel=loss_kernel,
            dim=1,
            inputs=[
                self.trajectory.body_q,
                self.target_pos,
                self.trajectory.step_count.numpy()[0],
                self.loss,
            ],
            device=self.solver.model.device,
        )
        return self.loss

    def train(self, iterations=50, learning_rate=1.0):
        # Setup Viewer for continuous history
        try:
            from newton.viewer import ViewerUSD
            viewer = ViewerUSD("optimization_history.usd", fps=60)
            viewer.set_model(self.model)
            recording = True
            print("Visualization enabled: recording history to 'optimization_history.usd'")
        except ImportError:
            recording = False
            print("Visualization disabled: usd-core not installed")

        print(f"Starting optimization with {type(self.engine_config).__name__}...")
        opt = wp.optim.SGD([self.param_vel], lr=learning_rate)
        steps = 60
        
        # Temp state for rendering
        render_state = self.model.state()
        frame_offset = 0

        for i in range(iterations):
            self.loss.zero_()
            tape = wp.Tape()

            # --- Phase 1: Initialization (Recorded) ---
            with tape:
                init_qd = wp.zeros((1, 1), dtype=wp.spatial_vector, requires_grad=True)
                wp.launch(
                    kernel=map_vel_kernel,
                    dim=1,
                    inputs=[self.param_vel, init_qd],
                    device=self.solver.model.device,
                )

            # --- Phase 2: Simulation (Manages its own recording) ---
            self.forward(steps=steps, tape=tape, qd=init_qd)

            # --- Phase 3: Loss (Recorded) ---
            with tape:
                l = self.compute_loss()

            # --- Phase 4: Backward ---
            tape.backward(l)

            if i % 5 == 0:
                print(f"Iter {i}: Loss={l.numpy()[0]:.4f} | Vel={self.param_vel.numpy()[0]}")

            # --- Record to Viewer ---
            if recording:
                num_steps = self.trajectory.step_count.numpy()[0]
                for f in range(num_steps):
                    time = (frame_offset + f) * self.effective_timestep
                    viewer.begin_frame(time)
                    
                    wp.launch(
                        kernel=flatten_slice_kernel,
                        dim=(self.model.num_worlds, self.model.body_count),
                        inputs=[
                            self.trajectory.body_q,
                            f,
                            render_state.body_q,
                            self.model.body_count,
                            self.model.num_worlds
                        ],
                        device=self.solver.model.device
                    )
                    viewer.log_state(render_state)
                    viewer.end_frame()
                
                frame_offset += num_steps

            opt.step([self.param_vel.grad])
            self.param_vel.grad.zero_()
            tape.zero()
        
        if recording:
            viewer.close()


def main():
    sim = BallBounceOptimizer(engine_type="semi_implicit")
    sim.train(iterations=20)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
