import numpy as np
import warp as wp
from axion.logging import HDF5Logger
from axion.logging import HDF5Reader


# Assume your struct and kernel definitions are available
@wp.struct
class SpatialInertia:
    m: wp.float32
    inertia: wp.mat33


@wp.kernel
def assemble_spatial_inertia_kernel(
    mass: wp.array(dtype=wp.float32),
    inertia: wp.array(dtype=wp.mat33),
    generalized_mass: wp.array(dtype=SpatialInertia),
):
    tid = wp.tid()
    generalized_mass[tid] = SpatialInertia(mass[tid], inertia[tid])


# And your HDF5Reader is available
# from hdf5_reader import HDF5Reader

# --- Setup ---
wp.init()
device = "cuda"
log_file = "simulation_simple_log.h5"
num_elements = 5

# --- Generate Sample Data ---
np_mass = np.arange(num_elements, dtype=np.float32)
np_inertia = np.array([np.eye(3) * (i + 1) for i in range(num_elements)], dtype=np.float32)
wp_mass = wp.array(np_mass, device=device)
wp_inertia = wp.array(np_inertia, dtype=wp.mat33, device=device)

# Create the struct array on the GPU
generalized_mass_arr = wp.empty(num_elements, dtype=SpatialInertia, device=device)
wp.launch(
    kernel=assemble_spatial_inertia_kernel,
    dim=num_elements,
    inputs=[wp_mass, wp_inertia],
    outputs=[generalized_mass_arr],
    device=device,
)

# --- Logging Phase ---
print("\n--- Logging Data ---")
with HDF5Logger(log_file, "w") as logger:
    with logger.scope("simulation_data"):
        # Log a regular primitive array
        logger.log_wp_dataset("body_masses", wp_mass)

        # Log the custom struct array using the dedicated method
        logger.log_struct_array(
            "generalized_mass",
            generalized_mass_arr,
            attributes={"description": "Spatial inertia of each body"},
        )

print("-" * 20)

# --- Reading Phase ---
print("\n--- Reading Data ---")
with HDF5Reader(log_file) as reader:
    print("File structure:")
    reader.print_tree()
