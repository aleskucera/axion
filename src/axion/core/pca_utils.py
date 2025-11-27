from __future__ import annotations

import torch
import warp as wp
from axion.constraints import batch_contact_residual_kernel
from axion.constraints import batch_friction_residual_kernel
from axion.constraints import batch_joint_residual_kernel
from axion.constraints import batch_unconstrained_dynamics_kernel
from newton import Model

from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def copy_h_to_history(
    h: wp.array(dtype=wp.float32),
    h_history: wp.array(dtype=wp.float32),
):
    world_idx, h_idx = wp.tid()

    h_history[world_idx, h_idx] = h[world_idx, h_idx]


@wp.kernel
def copy_body_u_to_history(
    body_u: wp.array(dtype=wp.spatial_vector),
    body_u_history: wp.array(dtype=wp.spatial_vector),
):
    world_idx, body_idx = wp.tid()

    body_u_history[world_idx, body_idx] = body_u[world_idx, body_idx]


@wp.kernel
def copy_body_lambda_to_history(
    body_lambda: wp.array(dtype=wp.float32),
    body_lambda_history: wp.array(dtype=wp.float32),
):
    world_idx, constraint_idx = wp.tid()

    body_lambda_history[world_idx, constraint_idx] = body_lambda[world_idx, constraint_idx]


def copy_state_to_history(
    newton_iteration: int,
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    # Evaluate residual for unconstrained dynamics
    wp.launch(
        kernel=copy_h_to_history,
        dim=(config.newton_iters, dims.N_w, dims.N_u + dims.N_c),
        inputs=[
            data._h_history[newton_iteration, :, :],
        ],
        outputs=[data._h],
        device=device,
    )
    wp.launch(
        kernel=copy_body_u_to_history,
        dim=(config.newton_iters, dims.N_w, dims.N_b),
        inputs=[
            data.body_u_history[newton_iteration, :, :],
        ],
        outputs=[data.body_u],
        device=device,
    )
    wp.launch(
        kernel=copy_body_lambda_to_history,
        dim=(config.newton_iters, dims.N_w, dims.N_c),
        inputs=[
            data._body_lambda[newton_iteration, :, :],
        ],
        outputs=[data._body_lambda],
        device=device,
    )


def perform_pca(
    trajectory_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs PCA on the given trajectory data to find the top two principal directions.

    These directions represent the axes of greatest variance in the data, which for an
    optimization trajectory, correspond to the most significant movements of the solver.

    Args:
        trajectory_data: A (num_steps, feature_dim) tensor of optimization variable states (x).

    Returns:
        A tuple containing:
        - v1 (torch.Tensor): The first principal component (direction of max variance).
        - v2 (torch.Tensor): The second principal component.
        - S (torch.Tensor): The first two singular values, representing the magnitude
          (approx. standard deviation) of the data along the principal components.
    """
    # Center the data by subtracting the mean for PCA
    data_mean = trajectory_data.mean(dim=0)
    data_centered = trajectory_data - data_mean

    # Perform PCA via Singular Value Decomposition (SVD)
    try:
        # We only need the top 2 components for a 2D visualization
        k = min(2, data_centered.shape[1], trajectory_data.shape[0])
        _U, S, Vh = torch.linalg.svd(data_centered, full_matrices=False)
        # The principal components are the rows of Vh
        v1 = Vh[0]
        # If there's only one valid point or dimension, the second component is zero
        v2 = Vh[1] if k > 1 else torch.zeros_like(v1)
        S_out = S[:k]
    except torch.linalg.LinAlgError:
        # As a fallback, use random orthogonal directions if SVD fails.
        # This can happen with ill-conditioned or zero-variance data.
        print("SVD failed. Using random directions for visualization.")
        x_shape = data_centered.shape[1]
        v1 = torch.randn(x_shape, device=data_centered.device)
        v1 /= torch.linalg.norm(v1)
        v2 = torch.randn(x_shape, device=data_centered.device)
        v2 = v2 - v1 * torch.dot(v2, v1)  # Orthogonalize v2 w.r.t. v1
        v2 /= torch.linalg.norm(v2)
        S_out = torch.ones(2, device=data_centered.device)

    return v1, v2, S_out


def create_pca_grid_batch(
    x_center: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    S: torch.Tensor,
    grid_res: int,
    plot_range_scale: float,
    N_u: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates a batch of points on a 2D grid defined by the principal components.

    This grid forms the 2D plane on which the loss landscape will be evaluated. The plane is
    centered at `x_center` and spanned by the directions `v1` and `v2`.

    Args:
        x_center: The point in high-dimensional space to center the grid on (usually the final solution).
        v1: The first principal component vector.
        v2: The second principal component vector.
        S: The singular values. S[0] is used to set a data-driven scale for the grid.
        grid_res: The resolution of the grid (grid_res x grid_res).
        plot_range_scale: A multiplier for the grid size. Acts as a "zoom" factor.
        N_u: The number of degrees of freedom for generalized velocities, used for splitting x.

    Returns:
        A tuple containing the points on the grid and the grid coordinates.
    """
    device = x_center.device

    # Use the largest singular value (S[0]) to define a data-driven plot range.
    # S[0] represents the "spread" of the trajectory data along its main direction.
    # Multiplying by `plot_range_scale` allows us to "zoom" in or out.
    # A scale of 3.0 is like plotting +/- 3 standard deviations, which usually provides a good view.
    max_range = plot_range_scale * (S[0].item() if S.numel() > 0 and S[0].item() > 1e-6 else 1.0)

    # `alphas` are the coordinates along the v1 direction.
    alphas = torch.linspace(-max_range, max_range, grid_res, device=device)
    # `betas` are the coordinates along the v2 direction.
    # Note: The range for beta is intentionally smaller (10x smaller). This is because the
    # variance along the second PC is often much less than the first. This creates a
    # rectangular plot that focuses on the most significant direction of change.
    betas = torch.linspace(-0.1 * max_range, 0.1 * max_range, grid_res, device=device)
    alpha_grid, beta_grid = torch.meshgrid(alphas, betas, indexing="ij")

    # Generate grid points in high-dimensional space using the 2D grid coordinates:
    # x_grid_point = x_center + alpha * v1 + beta * v2
    pca_batch_points = (
        x_center.unsqueeze(0)
        + alpha_grid.flatten().unsqueeze(1) * v1.unsqueeze(0)
        + beta_grid.flatten().unsqueeze(1) * v2.unsqueeze(0)
    )

    # Split the high-dimensional points back into u (velocity) and lambda (forces)
    pca_u = pca_batch_points[:, :N_u]
    pca_lambda = pca_batch_points[:, N_u:]
    return pca_u, pca_lambda, alphas, betas, alpha_grid, beta_grid


def compute_pca_batch_h_norm(
    model: Model,
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    """
    Computes the L2 norm of the residual vector `h` for a batch of points in parallel.

    This function orchestrates the launch of several Warp kernels, each calculating a
    part of the full residual vector `h = [h_d_v, h_j, h_n, h_f]`. Finally, it computes
    the norm of `h` for each point in the batch.

    Args:
        model: The newton.Model instance.
        data: The engine's data arrays, including the `pca_batch_*` arrays.
        config: The engine's configuration.
        dims: The engine's dimensions.

    Returns:
        A torch.Tensor of shape (B,) where B is the batch size, containing the L2 norm
        of the residual for each input point.
    """
    device = data.device

    data.pca_batch_h.full.zero_()

    B = data.pca_batch_body_u.shape[0]

    # Evaluate residual for unconstrained dynamics
    wp.launch(
        kernel=batch_unconstrained_dynamics_kernel,
        dim=(B, dims.N_b),
        inputs=[
            data.pca_batch_body_u,
            data.body_u_prev,
            data.body_f,
            data.body_M,
            data.dt,
            data.g_accel,
        ],
        outputs=[data.pca_batch_h.d_spatial],
        device=device,
    )

    # Evaluate residual for joint constraints
    wp.launch(
        kernel=batch_joint_residual_kernel,
        dim=(B, dims.N_j),
        inputs=[
            data.pca_batch_body_u,
            data.pca_batch_body_lambda.j,
            data.joint_constraint_data,
            data.dt,
            config.joint_stabilization_factor,
            config.joint_compliance,
        ],
        outputs=[
            data.pca_batch_h.d_spatial,
            data.pca_batch_h.j,
        ],
        device=device,
    )

    # Evaluate residual for normal contact constraints
    wp.launch(
        kernel=batch_contact_residual_kernel,
        dim=(B, dims.N_n),
        inputs=[
            data.pca_batch_body_u,
            data.body_u_prev,
            data.pca_batch_body_lambda.n,
            data.contact_interaction,
            data.body_M_inv,
            data.dt,
            config.contact_stabilization_factor,
            config.contact_fb_alpha,
            config.contact_fb_beta,
            config.contact_compliance,
        ],
        outputs=[
            data.pca_batch_h.d_spatial,
            data.pca_batch_h.n,
        ],
        device=device,
    )

    # Evaluate residual for friction constraints
    wp.launch(
        kernel=batch_friction_residual_kernel,
        dim=(B, dims.N_n),
        inputs=[
            data.pca_batch_body_u,
            data.pca_batch_body_lambda.f,
            data.body_lambda_prev.f,
            data.body_lambda_prev.n,
            data.s_n_prev,
            data.contact_interaction,
            config.friction_fb_alpha,
            config.friction_fb_beta,
            config.friction_compliance,
        ],
        outputs=[
            data.pca_batch_h.d_spatial,
            data.pca_batch_h.f,
        ],
        device=device,
    )

    # Finally, compute the L2 norm of the full residual vector h for each item in the batch.
    torch_h = wp.to_torch(data.pca_batch_h.full)
    torch_h_norm = torch.norm(torch_h, dim=1)  # The "loss" for each point on the grid

    # Copy the result back to its warp array (optional, but good practice)
    h_norm = wp.from_torch(torch_h_norm)
    wp.copy(data.pca_batch_h_norm, h_norm.contiguous())
    return torch_h_norm
