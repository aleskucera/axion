import numpy as np
import warp as wp
from axion.logging import HDF5Logger
from axion.logging import HDF5Reader
from axion.types import joint_interaction_kernel    #TODO
from axion.types import JointInteraction

# --- Setup ---
wp.init()
device = "cuda"
log_file = "joint_interaction_log.h5"
num_joints = 4  # Example number of joints to simulate

# --- Allocate Input/Output Arrays ---
# (You would normally fill these with meaningful data from your simulation)
wp_body_q = wp.array(
    [wp.transform_identity() for _ in range(num_joints * 2)], dtype=wp.transform, device=device
)
wp_body_com = wp.zeros(num_joints * 2, dtype=wp.vec3, device=device)
wp_joint_type = wp.array([wp.sim.JOINT_REVOLUTE] * num_joints, dtype=wp.int32, device=device)
wp_joint_enabled = wp.ones(num_joints, dtype=wp.int32, device=device)
wp_joint_parent = wp.array([i for i in range(num_joints)], dtype=wp.int32, device=device)
wp_joint_child = wp.array(
    [i + num_joints for i in range(num_joints)], dtype=wp.int32, device=device
)
# ... and so on for all other input arrays ...
wp_joint_X_p = wp.array([wp.transform_identity()] * num_joints, dtype=wp.transform, device=device)
wp_joint_X_c = wp.array([wp.transform_identity()] * num_joints, dtype=wp.transform, device=device)
wp_joint_axis_start = wp.zeros(num_joints, dtype=wp.int32, device=device)
wp_joint_axis = wp.array([wp.vec3(0, 0, 1)] * num_joints, dtype=wp.vec3, device=device)
wp_joint_linear_compliance = wp.zeros(num_joints, dtype=wp.float32, device=device)
wp_joint_angular_compliance = wp.zeros(num_joints, dtype=wp.float32, device=device)

# The output array for our kernel
interactions_arr = wp.empty(num_joints, dtype=JointInteraction, device=device)

# --- Run Kernel ---
wp.launch(
    kernel=joint_interaction_kernel,
    dim=num_joints,
    inputs=[
        wp_body_q,
        wp_body_com,
        wp_joint_type,
        wp_joint_enabled,
        wp_joint_parent,
        wp_joint_child,
        wp_joint_X_p,
        wp_joint_X_c,
        wp_joint_axis_start,
        wp_joint_axis,
        wp_joint_linear_compliance,
        wp_joint_angular_compliance,
    ],
    outputs=[interactions_arr],
)
wp.synchronize()

# --- Logging Phase ---
print("\n--- Logging Data ---")
with HDF5Logger(log_file, "w") as logger:
    logger.log_struct_array(
        "joint_interactions",
        interactions_arr,
        attributes={"description": "Constraint kinematics for all joints"},
    )

print("-" * 20)

# --- Reading Phase ---
print("\n--- Reading Data ---")
with HDF5Reader(log_file) as reader:
    print("File structure:")
    reader.print_tree()
