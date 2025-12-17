from __future__ import annotations

import torch
import warp as wp
from axion.constraints import batch_friction_residual_kernel
from axion.constraints import batch_positional_contact_residual_kernel
from axion.constraints import batch_positional_joint_residual_kernel
from axion.constraints import batch_unconstrained_dynamics_kernel
from axion.constraints import batch_velocity_contact_residual_kernel
from axion.constraints import batch_velocity_joint_residual_kernel
from axion.core.batched_model import BatchedModel

from .engine_config import EngineConfig
from .engine_data import EngineArrays
from .engine_dims import EngineDimensions


@wp.kernel
def copy_grid_q_kernel(
    src: wp.array(dtype=wp.float32, ndim=3),
    dest: wp.array(dtype=wp.transform, ndim=3),
):
    grid_idx, world_idx, body_idx = wp.tid()

    # src shape: (GridSize, World, N_q) where N_q = Body * 7
    base_idx = body_idx * 7

    # Layout: pos(3), quat(4) -> [px, py, pz, qx, qy, qz, qw]
    pos = wp.vec3(
        src[grid_idx, world_idx, base_idx + 0],
        src[grid_idx, world_idx, base_idx + 1],
        src[grid_idx, world_idx, base_idx + 2],
    )
    rot = wp.quat(
        src[grid_idx, world_idx, base_idx + 3],
        src[grid_idx, world_idx, base_idx + 4],
        src[grid_idx, world_idx, base_idx + 5],
        src[grid_idx, world_idx, base_idx + 6],
    )
    dest[grid_idx, world_idx, body_idx] = wp.transform(pos, rot)


@wp.kernel
def copy_grid_u_kernel(
    src: wp.array(dtype=float, ndim=3),  # Shape: (Grid, World, Body*6)
    dst: wp.array(dtype=wp.spatial_vector, ndim=3),  # Shape: (Grid, World, Body)
):
    g, w, b = wp.tid()

    # Each body has 6 DOFs in the flat source array
    base_idx = b * 6

    # Manually construct the spatial vector from the 6 floats
    # Warp's spatial_vector is (w_x, w_y, w_z, v_x, v_y, v_z)
    val = wp.spatial_vector(
        src[g, w, base_idx + 0],
        src[g, w, base_idx + 1],
        src[g, w, base_idx + 2],
        src[g, w, base_idx + 3],
        src[g, w, base_idx + 4],
        src[g, w, base_idx + 5],
    )

    dst[g, w, b] = val


@wp.kernel
def copy_grid_lambda_kernel(
    src: wp.array(dtype=float, ndim=3),  # Shape: (Grid, World, Constraints)
    dst: wp.array(dtype=float, ndim=3),  # Shape: (Grid, World, Constraints)
):
    g, w, c = wp.tid()
    dst[g, w, c] = src[g, w, c]


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
    Performs PCA on the given trajectory data.

    Args:
        trajectory_data: Input tensor.
          Expected shape: (Steps, Worlds, Dofs) or (Steps, Dofs).

    Returns:
        v1, v2: Principal components (Worlds, Dofs)
        S: Singular values or variance info (Worlds, 2)
    """

    # Handle the 3D case: (Steps, Worlds, Dofs)
    # We want to perform PCA over the 'Steps' dimension, independently for each 'World'.
    if trajectory_data.ndim == 3:
        # Permute to (Worlds, Steps, Dofs) for batch processing
        data = trajectory_data.permute(1, 0, 2)
    else:
        # Assume (Steps, Dofs), add a batch dimension -> (1, Steps, Dofs)
        data = trajectory_data.unsqueeze(0)

    # Center data: mean over Steps (dim 1)
    data_mean = data.mean(dim=1, keepdim=True)
    data_centered = data - data_mean

    # Perform Batched SVD
    # Input: (Worlds, Steps, Dofs)
    # Output Vh: (Worlds, Dofs, Dofs)
    try:
        k = min(2, data_centered.shape[1], data_centered.shape[2])
        _U, S, Vh = torch.linalg.svd(data_centered, full_matrices=False)

        # Principal components are rows of Vh
        v1 = Vh[:, 0, :]  # (Worlds, Dofs)
        v2 = Vh[:, 1, :] if k > 1 else torch.zeros_like(v1)
        S_out = S[:, :2]  # (Worlds, 2)

    except torch.linalg.LinAlgError:
        print("SVD failed. Using random directions.")
        worlds, _, dofs = data.shape
        v1 = torch.randn(worlds, dofs, device=data.device)
        v1 = torch.nn.functional.normalize(v1, dim=1)
        v2 = torch.randn(worlds, dofs, device=data.device)
        # Orthogonalize
        dot = (v2 * v1).sum(dim=1, keepdim=True)
        v2 = v2 - v1 * dot
        v2 = torch.nn.functional.normalize(v2, dim=1)
        S_out = torch.ones(worlds, 2, device=data.device)

    return v1, v2, S_out


def prepare_trajectory_data(
    q_history: torch.Tensor,  # Shape: (T, N_w, N_b, 7)
    u_history: torch.Tensor,  # Shape: (T, N_w, N_b, 6)
    lambda_history: torch.Tensor,  # Shape: (T, N_w, N_c)
) -> torch.Tensor:
    """
    Flattens and concatenates q, u, and lambda into a single feature vector per step.
    """
    T = q_history.shape[0]

    # Flatten spatial/body dimensions
    # q: (T, N_w * N_b * 7)
    q_flat = q_history.reshape(T, -1)
    # u: (T, N_w * N_b * 6)
    u_flat = u_history.reshape(T, -1)
    # lambda: (T, N_w * N_c)
    lambda_flat = lambda_history.reshape(T, -1)

    # Concatenate: [q, u, lambda]
    return torch.cat([q_flat, u_flat, lambda_flat], dim=1)


def create_pca_grid_batch(
    x_center: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    S: torch.Tensor,
    grid_res: int,
    plot_range_scale: float,
    dims: EngineDimensions,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates a batch of points on a 2D grid defined by the principal components.
    Handles x = [q, u, lambda].

    Args:
        S: Tensor of shape (Worlds, 2) containing bounds or singular values.
    """
    device = x_center.device

    # --- 1. Determine Plot Range ---
    # S is (Worlds, 2). S[:, 0] is X-range, S[:, 1] is Y-range.
    # We take the maximum range across all worlds to ensure the grid covers everything.
    # If S was generated by singular values, S[:, 1] might be small, so we handle that.

    if S.dim() == 2:
        max_bound_x = S[:, 0].max().item()
        # Use provided Y-bound if available and significant, otherwise default to 10% of X
        max_bound_y = S[:, 1].max().item()
        if max_bound_y < 1e-6:
            max_bound_y = 0.1 * max_bound_x
    else:
        # Fallback for 1D S
        max_bound_x = S[0].item()
        max_bound_y = 0.1 * max_bound_x

    range_x = plot_range_scale * (max_bound_x if max_bound_x > 1e-6 else 1.0)
    range_y = plot_range_scale * (max_bound_y if max_bound_y > 1e-6 else 1.0)

    # --- 2. Generate Grid Coordinates ---
    alphas = torch.linspace(-range_x, range_x, grid_res, device=device)
    betas = torch.linspace(-range_y, range_y, grid_res, device=device)
    alpha_grid, beta_grid = torch.meshgrid(alphas, betas, indexing="ij")

    # --- 3. Generate High-Dimensional Points ---
    # x_center: (Worlds, Dofs)
    # v1, v2: (Worlds, Dofs)
    # alpha_grid: (Grid, Grid) -> flatten -> (N_points)

    # We want output: (N_points, Worlds, Dofs)
    # Broadcasting: (1, Worlds, Dofs) + (N, 1, 1) * (1, Worlds, Dofs)

    pca_batch_points = (
        x_center.unsqueeze(0)
        + alpha_grid.flatten().view(-1, 1, 1) * v1.unsqueeze(0)
        + beta_grid.flatten().view(-1, 1, 1) * v2.unsqueeze(0)
    )

    # --- 4. Split and Reshape [q, u, lambda] ---
    # Calculate sizes
    num_bodies = dims.N_b  # Per world
    dim_q = num_bodies * 7
    dim_u = num_bodies * 6
    # dim_lambda is the remainder

    # Slice
    pca_q_flat = pca_batch_points[:, :, :dim_q]
    pca_u_flat = pca_batch_points[:, :, dim_q : dim_q + dim_u]
    pca_lambda_flat = pca_batch_points[:, :, dim_q + dim_u :]

    # --- 5. Normalize Quaternions in Q ---
    # pca_q_flat is (Points, Worlds, Nb*7)
    # Reshape to access quaternions: (Points, Worlds, Nb, 7)
    pca_q_reshaped = pca_q_flat.view(-1, dims.N_w, dims.N_b, 7)

    pos = pca_q_reshaped[..., :3]
    rot = pca_q_reshaped[..., 3:]

    rot = torch.nn.functional.normalize(rot, p=2, dim=-1)

    # Reassemble and Flatten for return
    pca_q_final = torch.cat([pos, rot], dim=-1)  # (P, W, B, 7)

    # Reshape U and Lambda for Warp kernels
    # pca_u: (P, W, B, 6) -> Flatten last dim -> (P, W, B*6) for torch return?
    # Actually the kernel expects (P, W, B) spatial vectors.
    # Torch -> Warp conversion usually takes flat lists or structured.
    # Let's keep them as flat floats in Torch and let the Kernel loader handle it?
    # No, we have specific Copy kernels now.

    # The copy kernels expect:
    # src: (GridSize, World, Components) (float32 array)
    # dest: (GridSize, World, Body) (spatial_vector/transform array)

    # So we return float tensors of shape (N_points, N_worlds, N_components)
    return (
        pca_q_final.reshape(-1, dims.N_w, dim_q),
        pca_u_flat,
        pca_lambda_flat,
        alphas,
        betas,
        alpha_grid,
        beta_grid,
    )


def compute_pca_batch_h_norm(
    model: BatchedModel,
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
            data.pca_batch_body_q,
            data.pca_batch_body_u,
            data.body_u_prev,
            data.body_f,
            model.body_mass,
            model.body_inertia,
            data.dt,
            data.g_accel,
        ],
        outputs=[data.pca_batch_h.d_spatial],
        device=device,
    )

    if config.joint_constraint_level == "pos":
        # wp.launch(
        #     kernel=batch_positional_joint_residual_kernel,
        #     dim=(B, dims.N_w, dims.N_j),
        #     inputs=[
        #         data.pca_batch_body_u,
        #         data.pca_batch_body_lambda.j,
        #         data.joint_constraint_data,
        #         data.dt,
        #         config.joint_stabilization_factor,
        #         config.joint_compliance,
        #     ],
        #     outputs=[
        #         data.pca_batch_h.d_spatial,
        #         data.pca_batch_h.c.j,
        #     ],
        #     device=device,
        # )
        wp.launch(
            kernel=batch_positional_joint_residual_kernel,
            dim=(B, dims.N_w, dims.joint_count),
            inputs=[
                data.pca_batch_body_q,
                data.pca_batch_body_lambda.j,
                model.body_com,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_qd_start,
                data.joint_constraint_offsets,
                data.dt,
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
                data.pca_batch_body_q,
                data.pca_batch_body_u,
                data.body_u_prev,
                data.pca_batch_body_lambda.n,
                data.contact_interaction,
                model.body_mass,
                model.body_inertia,
                data.dt,
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
