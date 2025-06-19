import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from axion.nsn_engine import linearize_system

# --- Constants ---
PERTURBATION = 1e-4  # Small value for finite difference
ERROR_ABSOLUTE_TOLERANCE = 1e-8  # For combined error calculation


def setup_test_scene(device):
    """
    Creates a simple simulation Model with a dynamic box and a static ground plane.
    """
    builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))

    # Add a dynamic box with some initial rotation
    builder.add_body(
        origin=wp.transform(
            (0.0, 0.45, 0.0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.0)
        ),
        name="box",
    )
    builder.add_shape_sphere(body=0, radius=0.5, density=10.0)

    model = builder.finalize(device=device)
    model.ground = True  # Ensures contact with ground plane is handled

    # Initialize states
    state_in = model.state()
    state_out = model.state()

    # Set some initial velocity for the box to make dynamics non-trivial
    initial_qd = np.array([[0.01, 0.02, 0.03, 0.0, 0.0, -0.1]], dtype=np.float32)
    state_in.body_qd.assign(
        wp.array(initial_qd, dtype=wp.spatial_vector, device=device)
    )

    state_out.body_qd.assign(state_in.body_qd)
    state_out.body_q.assign(state_in.body_q)

    # Manually create a single contact for testing purposes
    # This ensures the contact part of the jacobian is populated
    model.rigid_contact_count.assign(wp.array([1], dtype=wp.int32, device=device))
    model.rigid_contact_point0.assign(
        wp.array([[0.0, 0.0, -0.5]], dtype=wp.vec3, device=device)
    )
    model.rigid_contact_point1.assign(
        wp.array([[0.0, 0.0, -0.05]], dtype=wp.vec3, device=device)
    )
    model.rigid_contact_normal.assign(
        wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3, device=device)
    )
    model.rigid_contact_shape0.assign(
        wp.array([0], dtype=wp.int32, device=device)
    )  # box shape
    model.rigid_contact_shape1.assign(
        wp.array([-1], dtype=wp.int32, device=device)
    )  # ground plane

    return model, state_in, state_out


def compute_residual(model, state_in, state_out, dt, lambda_n):
    """
    Wrapper function to compute the residual vector of the system.
    This calls linearize_system but only returns the residual part.
    """
    B = model.body_count
    C = model.rigid_contact_max

    neg_res = wp.zeros((6 * B + C,), dtype=wp.float32, device=model.device)
    jacobian_temp = wp.zeros(
        (6 * B + C, 6 * B + C), dtype=wp.float32, device=model.device
    )

    linearize_system(model, state_in, state_out, dt, lambda_n, neg_res, jacobian_temp)

    return -neg_res.numpy()


def compute_jacobian_fd(model, state_in, state_out, dt, lambda_n, h=PERTURBATION):
    """
    Computes the Jacobian matrix using the central finite difference method.
    """
    B = model.body_count
    C = model.rigid_contact_max
    num_vars = 6 * B + C

    print(f"Computing finite difference Jacobian for {num_vars} variables...")
    jacobian_fd = np.zeros((num_vars, num_vars))

    for j in range(num_vars):
        state_out_plus = model.state(requires_grad=False)
        state_out_minus = model.state(requires_grad=False)
        state_out_plus.body_qd.assign(state_out.body_qd)
        state_out_minus.body_qd.assign(state_out.body_qd)

        lambda_n_plus = wp.clone(lambda_n)
        lambda_n_minus = wp.clone(lambda_n)

        h_vec = np.zeros(num_vars)
        h_vec[j] = h

        if j < 6 * B:
            body_idx, comp_idx = j // 6, j % 6
            state_out_plus.body_qd.numpy()[body_idx][comp_idx] += h
            state_out_minus.body_qd.numpy()[body_idx][comp_idx] -= h
        else:
            lambda_idx = j - 6 * B
            lambda_n_plus.numpy()[lambda_idx] += h
            lambda_n_minus.numpy()[lambda_idx] -= h

        res_plus = compute_residual(model, state_in, state_out_plus, dt, lambda_n_plus)
        res_minus = compute_residual(
            model, state_in, state_out_minus, dt, lambda_n_minus
        )

        jacobian_fd[:, j] = (res_plus - res_minus) / (2.0 * h)

        if (j + 1) % 10 == 0 or j == num_vars - 1:
            print(f"  ... processed column {j + 1}/{num_vars}")

    return jacobian_fd


def compute_combined_error(J_analytic, J_numerical, atol=ERROR_ABSOLUTE_TOLERANCE):
    """
    Computes a combined absolute/relative error metric:
    error = |analytic - numerical| / (atol + |numerical|)
    """
    abs_diff = np.abs(J_analytic - J_numerical)
    # Use the numerical jacobian as the reference for scaling
    denominator = atol + np.abs(J_numerical)
    return abs_diff / denominator


def visualize_comparison(J_analytic, J_numerical, J_error):
    """
    Generates a matplotlib plot comparing the analytical, numerical, and error matrices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Determine common color limits for the first two plots
    vmax = max(np.max(np.abs(J_analytic)), np.max(np.abs(J_numerical)))
    vmin = -vmax

    # Plot Analytical Jacobian
    im0 = axes[0].imshow(J_analytic, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Analytical Jacobian")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Numerical Jacobian
    im1 = axes[1].imshow(J_numerical, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Numerical (Finite Diff.) Jacobian")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot Combined Error using a logarithmic scale
    max_error = np.max(J_error)
    im2 = axes[2].imshow(
        J_error,
        cmap="magma",
        norm=colors.LogNorm(
            vmin=max(1e-9, np.min(J_error[J_error > 0])), vmax=max_error
        ),
    )
    axes[2].set_title(f"Combined Error (Max: {max_error:.2e})")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("Jacobian Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("jacobian_comparison.png", dpi=300)
    print("\nSaved visualization to jacobian_comparison.png")
    # plt.show()


if __name__ == "__main__":
    wp.init()
    device = wp.get_device()
    print(f"Running on device: {device}")

    # --- 1. Setup ---
    model, state_in, state_out = setup_test_scene(device)
    dt = 1.0 / 60.0
    B = model.body_count
    C = model.rigid_contact_max

    # Initial guess for lambda (contact forces)
    lambda_n = wp.zeros((C,), dtype=wp.float32, device=device)

    # --- 2. Compute Analytical Jacobian ---
    print("\nComputing analytical Jacobian...")
    neg_res_analytic = wp.zeros((6 * B + C,), dtype=wp.float32, device=device)
    jacobian_analytic = wp.zeros(
        (6 * B + C, 6 * B + C), dtype=wp.float32, device=device
    )
    linearize_system(
        model, state_in, state_out, dt, lambda_n, neg_res_analytic, jacobian_analytic
    )
    J_analytic_np = jacobian_analytic.numpy()
    print("Analytical Jacobian computation complete.")

    # --- 3. Compute Numerical Jacobian ---
    J_numerical_np = compute_jacobian_fd(model, state_in, state_out, dt, lambda_n)
    print("Numerical Jacobian computation complete.")

    # --- 4. Compare and Report ---
    J_error = compute_combined_error(J_analytic_np, J_numerical_np)
    max_error = np.max(J_error)
    mean_error = np.mean(J_error)

    print("\n--- Jacobian Verification Results ---")
    print(f"Max Combined Error: {max_error:.6e}")
    print(f"Mean Combined Error: {mean_error:.6e}")

    # An error less than 1e-3 is generally a good sign.
    if max_error < 1e-3:
        print(
            "✅ Test Passed: The analytical Jacobian closely matches the numerical one."
        )
    else:
        print(
            "❌ Test Failed: Significant deviation found between analytical and numerical Jacobians."
        )

    print("-------------------------------------\n")

    # --- 5. Visualize ---
    visualize_comparison(J_analytic_np, J_numerical_np, J_error)
