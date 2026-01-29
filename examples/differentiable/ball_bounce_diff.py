import numpy as np
import warp as wp
import warp.optim
from newton import Model
from newton import ModelBuilder

from axion import AxionEngineConfig
from axion import DifferentiableSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import RenderingConfig
from axion import SemiImplicitEngineConfig
from axion import SimulationConfig


@wp.kernel
def loss_kernel(
    body_q_traj: wp.array(dtype=wp.transform, ndim=3),
    target_pos: wp.vec3,
    step_count: int,
    loss: wp.array(dtype=float),
):
    # Loss is distance of the LAST step to the target
    last_step = step_count - 1
    
    # Assuming single world (index 0) and single body (index 0)
    pos_xform = body_q_traj[last_step, 0, 0]
    pos = wp.transform_get_translation(pos_xform)
    
    delta = pos - target_pos
    loss[0] = wp.dot(delta, delta)


class BallBounceOptimizer(DifferentiableSimulator):
    def __init__(self, engine_type="semi_implicit"):
        sim_config = SimulationConfig(
            duration_seconds=1.0, 
            target_timestep_seconds=1.0 / 60.0
        )
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
        # We start with a guess for the initial velocity
        self.init_vel = wp.vec3(0.0, 5.0, -5.0)
        self.param_vel = wp.array([self.init_vel], dtype=wp.vec3, requires_grad=True)
        
        # We need a full-sized array to pass to reset_state, matching model.body_qd
        # Shape: (num_worlds, num_bodies) -> (1, 1) spatial vectors or vec3?
        # Newton particles use particle_qd (vec3), rigid bodies use body_qd (spatial).
        # Axion/BaseSimulator unifies this. BaseSimulator uses body_qd (spatial).
        # But ModelBuilder.add_particle creates a particle.
        # Let's check BaseSimulator._copy_state again. It uses body_q/body_qd.
        
        # NOTE: If we use particles in Newton, they are stored in particle_q/qd.
        # But BaseSimulator copies body_q/qd.
        # We need to verify if Model stores particles in body arrays too.
        # Yes, Newton unifies storage for some solvers, but separate for others.
        # For this example, let's use a Rigid Body Sphere to be safe with spatial vectors.

    def build_model(self) -> Model:
        builder = ModelBuilder()
        
        # Using a rigid body sphere instead of a particle to match Axion's spatial vector expectations
        builder.add_body(
            xform=wp.transform(wp.vec3(0.0, -0.5, 1.0), wp.quat_identity()),
            mass=1.0
        )
        builder.add_shape_sphere(body=0, radius=0.1)

        builder.add_ground_plane()
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()),
            hx=1.0, hy=0.25, hz=1.0
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
                self.loss
            ],
            device=self.solver.model.device
        )
        return self.loss

    def train(self, iterations=50, learning_rate=1.0):
        print(f"Starting optimization with {type(self.engine_config).__name__}...")
        
        opt = wp.optim.SGD([self.param_vel], lr=learning_rate)
        steps = 60
        
        for i in range(iterations):
            self.loss.zero_()
            
            # 1. Prepare Initial Velocity Array
            # We map our 1 optimization var (vec3) to the spatial vector (6D) required by the engine.
            # Spatial velocity w = [omega, v]. We set v.
            # We need to do this INSIDE the tape if we want gradients to flow back to param_vel.
            
            tape = wp.Tape()
            with tape:
                # Construct the full initial state array
                # (1 world, 1 body)
                init_qd = wp.zeros((1, 1), dtype=wp.spatial_vector, requires_grad=True)
                
                # Assign our parameter to the linear velocity part
                # Custom kernel or map needed? 
                # Warp arrays support slicing.
                # init_qd[0, 0] = wp.spatial_vector(0,0,0, vx, vy, vz)
                
                # Let's use a tiny kernel to map vec3 -> spatial vector to be clean
                wp.launch(
                    kernel=map_vel_kernel,
                    dim=1,
                    inputs=[self.param_vel, init_qd],
                    device=self.solver.model.device
                )

                # 2. Run Forward Simulation
                # Passing our tracked 'init_qd' to reset the state
                self.forward(steps=steps, record_tape=False, qd=init_qd)
                
                # 3. Compute Loss
                l = self.compute_loss()
            
            # 4. Backward Pass
            tape.backward(l)
            
            # 5. Step Optimizer
            if i % 5 == 0:
                print(f"Iter {i}: Loss={l.numpy()[0]:.4f} | Vel={self.param_vel.numpy()[0]}")
                
            opt.step([self.param_vel.grad])
            
            # Zero gradients
            self.param_vel.grad.zero_()

@wp.kernel
def map_vel_kernel(v_in: wp.array(dtype=wp.vec3), v_out: wp.array(dtype=wp.spatial_vector, ndim=2)):
    # Maps vec3 to the linear part of spatial vector (body 0, world 0)
    v = v_in[0]
    v_out[0, 0] = wp.spatial_vector(0.0, 0.0, 0.0, v[0], v[1], v[2])


def main():
    # Run with Semi-Implicit (Newton)
    sim = BallBounceOptimizer(engine_type="semi_implicit")
    sim.train(iterations=20)

    # Run with Axion (if desired)
    # sim_axion = BallBounceOptimizer(engine_type="axion")
    # sim_axion.train(iterations=20)

if __name__ == "__main__":
    main()