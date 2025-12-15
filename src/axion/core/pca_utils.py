from __future__ import annotations

import torch
import warp as wp
from axion.constraints import batch_friction_residual_kernel
from axion.constraints import batch_positional_contact_residual_kernel
from axion.constraints import batch_positional_joint_residual_kernel
from axion.constraints import batch_unconstrained_dynamics_kernel
from axion.constraints import batch_velocity_contact_residual_kernel
from axion.constraints import batch_velocity_joint_residual_kernel
from newton import Model

from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def copy_h_to_history(
    h: wp.array(dtype=wp.float32, ndim=2),
    h_history: wp.array(dtype=wp.float32, ndim=2),
):
    world_idx, h_idx = wp.tid()
    h_history[world_idx, h_idx] = h[world_idx, h_idx]


@wp.kernel
def copy_body_q_to_history(
    body_q: wp.array(dtype=wp.transform, ndim=2),
    body_q_history: wp.array(dtype=wp.transform, ndim=2),
):
    world_idx, body_idx = wp.tid()
    body_q_history[world_idx, body_idx] = body_q[world_idx, body_idx]


@wp.kernel
def copy_body_u_to_history(
    body_u: wp.array(dtype=wp.spatial_vector, ndim=2),
    body_u_history: wp.array(dtype=wp.spatial_vector, ndim=2),
):
    world_idx, body_idx = wp.tid()
    body_u_history[world_idx, body_idx] = body_u[world_idx, body_idx]


@wp.kernel
def copy_body_lambda_to_history(
    body_lambda: wp.array(dtype=wp.float32, ndim=2),
    body_lambda_history: wp.array(dtype=wp.float32, ndim=2),
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

    wp.launch(
        kernel=copy_h_to_history,
        dim=(dims.N_w, dims.N_u + dims.N_c),
        inputs=[data._h],
        outputs=[data._h_history[newton_iteration, :, :]],
        device=device,
    )
    wp.launch(
        kernel=copy_body_q_to_history,
        dim=(dims.N_w, dims.N_b),
        inputs=[data.body_q],
        outputs=[data.body_q_history[newton_iteration, :, :]],
        device=device,
    )
    wp.launch(
        kernel=copy_body_u_to_history,
        dim=(dims.N_w, dims.N_b),
        inputs=[data.body_u],
        outputs=[data.body_u_history[newton_iteration, :, :]],
        device=device,
    )
    wp.launch(
        kernel=copy_body_lambda_to_history,
        dim=(dims.N_w, dims.N_c),
        inputs=[data._body_lambda],
        outputs=[data._body_lambda_history[newton_iteration, :, :]],
        device=device,
    )


def perform_pca(
    trajectory_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs PCA on the trajectory data for ALL worlds in parallel.
    Args:
        trajectory_data: (Steps, Worlds, Dofs)
    Returns:
        v1: (Worlds, Dofs)
        v2: (Worlds, Dofs)
        S:  (Worlds, 2)
    """
    # Mean over steps
    data_mean = trajectory_data.mean(dim=0)
    data_centered = trajectory_data - data_mean

    # Permute to (Worlds, Steps, Dofs) for batched SVD
    data_centered_permuted = data_centered.permute(1, 0, 2)

    num_worlds = data_centered_permuted.shape[0]
    feature_dim = data_centered_permuted.shape[2]

    try:
        _U, S, Vh = torch.linalg.svd(data_centered_permuted, full_matrices=False)
        # Vh is (Worlds, Dofs, Dofs) - rows are components
        v1 = Vh[:, 0, :]
        v2 = Vh[:, 1, :]
        S_out = S[:, :2]
    except torch.linalg.LinAlgError:
        print("SVD failed. Using random directions.")
        v1 = torch.randn((num_worlds, feature_dim), device=trajectory_data.device)
        v1 = torch.nn.functional.normalize(v1, dim=1)
        v2 = torch.randn((num_worlds, feature_dim), device=trajectory_data.device)
        dots = (v1 * v2).sum(dim=1, keepdim=True)
        v2 = v2 - v1 * dots
        v2 = torch.nn.functional.normalize(v2, dim=1)
        S_out = torch.ones((num_worlds, 2), device=trajectory_data.device)

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
    Creates a batch of points for ALL worlds simultaneously.

    Args:
        x_center: (N_w, Dofs)
        v1, v2: (N_w, Dofs)
        S: (N_w, 2)

    Returns:
        pca_u: (GridSize, N_w, N_u)
        pca_lambda: (GridSize, N_w, N_c)
        alphas: (N_w, GridRes) - Axis coordinates for each world
        betas: (N_w, GridRes)
    """
    device = x_center.device

    # 1. Determine plot range per world
    s0 = S[:, 0]  # (N_w,)
    s0 = torch.where(s0 > 1e-6, s0, torch.tensor(1.0, device=device))
    max_range = plot_range_scale * s0  # (N_w,)

    # 2. Create canonical normalized grid indices [-1, 1]
    # This grid shape is constant, but values are scaled per world later
    lin_norm = torch.linspace(-1.0, 1.0, grid_res, device=device)

    # 3. Create output axes for plotting (scaled)
    # alphas: (N_w, GridRes)
    alphas = max_range.unsqueeze(1) * lin_norm.unsqueeze(0)
    betas = (0.1 * max_range).unsqueeze(1) * lin_norm.unsqueeze(0)

    # 4. Create meshgrid coefficients
    # alpha_norm, beta_norm are (GridRes, GridRes)
    alpha_norm, beta_norm = torch.meshgrid(lin_norm, lin_norm, indexing="ij")

    # Flatten to (GridSize,)
    alpha_flat = alpha_norm.flatten()
    beta_flat = beta_norm.flatten()

    # 5. Compute coefficients for every point in every world
    # alpha_vals: (GridSize, N_w)
    alpha_vals = alpha_flat.unsqueeze(1) * max_range.unsqueeze(0)
    beta_vals = beta_flat.unsqueeze(1) * (0.1 * max_range.unsqueeze(0))

    # 6. Generate High-Dimensional Points
    # x_center: (1, N_w, Dofs)
    # v1: (1, N_w, Dofs)
    # alpha_vals: (GridSize, N_w, 1)
    pca_batch_points = (
        x_center.unsqueeze(0)
        + alpha_vals.unsqueeze(2) * v1.unsqueeze(0)
        + beta_vals.unsqueeze(2) * v2.unsqueeze(0)
    )

    pca_u = pca_batch_points[:, :, :N_u]
    pca_lambda = pca_batch_points[:, :, N_u:]

    return pca_u, pca_lambda, alphas, betas


def compute_pca_batch_h_norm(
    model: Model,
    data: EngineArrays,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device
    data.pca_batch_h.full.zero_()

    B = data.pca_batch_body_u.shape[0]  # Grid Size

    # Launch kernels over (GridSize, N_w, ...)
    # All kernels support batch dimensions naturally
    wp.launch(
        kernel=batch_unconstrained_dynamics_kernel,
        dim=(B, dims.N_w, dims.N_b),
        inputs=[
            data.pca_batch_body_u,
            data.body_u_prev,
            data.body_f,
            data.world_M,
            data.dt,
            data.g_accel,
        ],
        outputs=[data.pca_batch_h.d_spatial],
        device=device,
    )

    if config.joint_constraint_level == "pos":
        wp.launch(
            kernel=batch_positional_joint_residual_kernel,
            dim=(B, dims.N_w, dims.N_j),
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
                data.pca_batch_h.c.j,
            ],
            device=device,
        )
    elif config.joint_constraint_level == "vel":
        wp.launch(
            kernel=batch_velocity_joint_residual_kernel,
            dim=(B, dims.N_w, dims.N_j),
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
                data.pca_batch_h.c.j,
            ],
            device=device,
        )
    else:
        raise ValueError("Joint constraint level can be only 'pos' or 'vel'.")

    if config.contact_constraint_level == "pos":
        wp.launch(
            kernel=batch_positional_contact_residual_kernel,
            dim=(B, dims.N_w, dims.N_n),
            inputs=[
                data.pca_batch_body_u,
                data.body_u_prev,
                data.pca_batch_body_lambda.n,
                data.contact_interaction,
                data.world_M,
                data.dt,
                config.contact_stabilization_factor,
                config.contact_fb_alpha,
                config.contact_fb_beta,
                config.contact_compliance,
            ],
            outputs=[
                data.pca_batch_h.d_spatial,
                data.pca_batch_h.c.n,
            ],
            device=device,
        )
    elif config.contact_constraint_level == "vel":
        wp.launch(
            kernel=batch_velocity_contact_residual_kernel,
            dim=(B, dims.N_w, dims.N_n),
            inputs=[
                data.pca_batch_body_u,
                data.body_u_prev,
                data.pca_batch_body_lambda.n,
                data.contact_interaction,
                data.world_M_inv,
                data.dt,
                config.contact_stabilization_factor,
                config.contact_fb_alpha,
                config.contact_fb_beta,
                config.contact_compliance,
            ],
            outputs=[
                data.pca_batch_h.d_spatial,
                data.pca_batch_h.c.n,
            ],
            device=device,
        )
    else:
        raise ValueError("Contact constraint level can be only 'pos' or 'vel'.")

    wp.launch(
        kernel=batch_friction_residual_kernel,
        dim=(B, dims.N_w, dims.N_n),
        inputs=[
            data.pca_batch_body_u,
            data.pca_batch_body_lambda.f,
            data.body_lambda_prev.f,
            data.body_lambda_prev.n,
            data.s_n_prev,
            data.contact_interaction,
            data.dt,
            config.friction_fb_alpha,
            config.friction_fb_beta,
            config.friction_compliance,
        ],
        outputs=[
            data.pca_batch_h.d_spatial,
            data.pca_batch_h.c.f,
        ],
        device=device,
    )

    # Compute Norm
    # data.pca_batch_h.full is (GridSize, N_w, Dofs)
    torch_h = wp.to_torch(data.pca_batch_h.full)
    torch_h_norm = torch.norm(torch_h, dim=-1)  # -> (GridSize, N_w)

    h_norm = wp.from_torch(torch_h_norm)
    wp.copy(data.pca_batch_h_norm, h_norm.contiguous())

    return torch_h_norm
