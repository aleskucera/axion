
import newton
import numpy as np
import pytest
import warp as wp
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_logger import EngineLogger
from axion.core.engine_logger import LoggingConfig
from axion.core.model_builder import AxionModelBuilder
from axion.core.control_utils import JointMode

wp.init()

def setup_test_engine():
    # Setup Engine
    config = AxionEngineConfig(
        joint_constraint_level="pos",
        joint_compliance=0.0,
        newton_iters=4,
        linear_iters=4,
    )
    logger = EngineLogger(LoggingConfig(enable_timing=False))
    logger.initialize_events(steps_per_segment=1, newton_iters=config.newton_iters)

    return config, logger

def create_wheel_model(joint_type="Revolute"):
    builder = AxionModelBuilder()

    # Create a wheel (box for simplicity, inertia matters more)
    link_1 = builder.add_link()
    # Heavier for inertia
    builder.add_shape_box(
        link_1, 
        hx=0.5, 
        hy=0.5, 
        hz=0.1, 
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0)
    )

    # Joint connected to world (parent=-1)
    # Z-axis rotation
    axis = wp.vec3(0.0, 0.0, 1.0)
    
    parent_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
    child_local_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())

    if joint_type == "Revolute":
        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_1,
            axis=axis,
            parent_xform=parent_local_xform,
            child_xform=child_local_xform
        )
    
    builder.add_articulation([j0], key="wheel_articulation")
    
    # Finalize model
    model = builder.finalize_replicated(num_worlds=1, gravity=0.0) # No gravity to isolate control
    return model, j0

def test_position_control():
    print("\n=== Testing Position Control ===")
    model, joint_id = create_wheel_model()
    config, logger = setup_test_engine()
    
    engine = None
    def init_state_fn(state_in, state_out, contacts, dt):
        engine.integrate_bodies(model, state_in, state_out, dt)

    engine = AxionEngine(model=model, init_state_fn=init_state_fn, logger=logger, config=config)
    
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.collide(state_in)
    dt = 0.0166 # 60Hz

    # Initialize FK
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    
    # Setup Control
    target_pos = np.pi / 2.0 # 90 degrees
    
    # Set Mode
    wp.copy(model.joint_dof_mode, wp.array(np.array([int(JointMode.TARGET_POSITION)], dtype=np.int32), dtype=wp.int32, device=model.device))
    
    # Set Gains (Tune these)
    # Critical damping for stability and convergence
    # I = 3.33. ke=100 -> omega=5.5.
    # kd_crit = 2 * I * omega = 36.4. Use 40.0.
    wp.copy(model.joint_target_ke, wp.array(np.array([100.0], dtype=np.float32), dtype=wp.float32, device=model.device))
    wp.copy(model.joint_target_kd, wp.array(np.array([40.0], dtype=np.float32), dtype=wp.float32, device=model.device))
    
    
    print("Running Position Control Simulation...")
    
    final_error = 0.0
    
    for step in range(150): # Run a bit longer
        # Clear external forces
        state_in.body_f.zero_()
        
        # Set target in control object
        wp.copy(control.joint_target, wp.array(np.array([target_pos], dtype=np.float32), dtype=wp.float32, device=model.device))
        
        engine.step(state_in, state_out, control, contacts, dt)
        
        # Swap states
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        
        # Let's rely on eval_ik to update model.joint_q for checking
        newton.eval_ik(model, state_in, model.joint_q, model.joint_qd)
        
        current_q = model.joint_q.numpy()[0]
        
        if step % 20 == 0 or step == 149:
            print(f"Step {step}: q={current_q:.4f}, target={target_pos:.4f}")
            
        final_error = abs(current_q - target_pos)

    print(f"Final Position Error: {final_error:.4f}")
    assert final_error < 0.01, f"Position control failed to converge. Error: {final_error}"

    print(f"Final Position Error: {final_error:.4f}")
    assert final_error < 0.1, f"Position control failed to converge. Error: {final_error}"

def test_velocity_control():
    print("\n=== Testing Velocity Control ===")
    model, joint_id = create_wheel_model()
    config, logger = setup_test_engine()
    
    engine = None
    def init_state_fn(state_in, state_out, contacts, dt):
        engine.integrate_bodies(model, state_in, state_out, dt)

    engine = AxionEngine(model=model, init_state_fn=init_state_fn, logger=logger, config=config)
    
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.collide(state_in)
    dt = 0.0166

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    
    target_vel = 5.0 # rad/s
    
    # Set Mode
    wp.copy(model.joint_dof_mode, wp.array(np.array([int(JointMode.TARGET_VELOCITY)], dtype=np.int32), dtype=wp.int32, device=model.device))
    
    # Set Gains
    # For Velocity mode:
    # err = target - qd
    # control_f = ke * err ...
    # So ke is the velocity gain.
    # ke=20.0 -> tau=I/ke = 3.33/20 = 0.16s. dt=0.016. Stable.
    wp.copy(model.joint_target_ke, wp.array(np.array([20.0], dtype=np.float32), dtype=wp.float32, device=model.device))
    wp.copy(model.joint_target_kd, wp.array(np.array([0.0], dtype=np.float32), dtype=wp.float32, device=model.device))
    
    print("Running Velocity Control Simulation...")
    
    final_error = 0.0
    
    for step in range(100):
        state_in.body_f.zero_()
        
        wp.copy(control.joint_target, wp.array(np.array([target_vel], dtype=np.float32), dtype=wp.float32, device=model.device))
        
        engine.step(state_in, state_out, control, contacts, dt)
        
        # Swap states
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        
        newton.eval_ik(model, state_in, model.joint_q, model.joint_qd)
        
        current_qd = model.joint_qd.numpy()[0]
        
        if step % 20 == 0 or step == 99:
            print(f"Step {step}: qd={current_qd:.4f}, target={target_vel:.4f}")
            
        final_error = abs(current_qd - target_vel)

    print(f"Final Velocity Error: {final_error:.4f}")
    assert final_error < 0.01, f"Velocity control failed to converge. Error: {final_error}" 

if __name__ == "__main__":
    test_position_control()
    test_velocity_control()
