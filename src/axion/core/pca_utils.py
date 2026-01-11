from __future__ import annotations

import torch
import warp as wp
from axion.constraints import batch_friction_residual_kernel
from axion.constraints import batch_positional_contact_residual_kernel
from axion.constraints import batch_positional_joint_residual_kernel
from axion.constraints import batch_unconstrained_dynamics_kernel
from axion.constraints import batch_velocity_contact_residual_kernel
from axion.constraints import batch_velocity_joint_residual_kernel

from .engine_config import EngineConfig
from .engine_data import EngineData
from .engine_dims import EngineDimensions
from .model import AxionModel


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
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device

    wp.launch(
        kernel=copy_h_to_history,
        dim=(dims.N_w, dims.N_u + dims.N_c),
        inputs=[data._h],
        outputs=[data.history._h_history[newton_iteration, :, :]],
        device=device,
    )
    wp.launch(
        kernel=copy_body_q_to_history,
        dim=(dims.N_w, dims.N_b),
        inputs=[data.body_q],
        outputs=[data.history.body_q_history[newton_iteration, :, :]],
        device=device,
    )
    wp.launch(
        kernel=copy_body_u_to_history,
        dim=(dims.N_w, dims.N_b),
        inputs=[data.body_u],
        outputs=[data.history.body_u_history[newton_iteration, :, :]],
        device=device,
    )
    wp.launch(
        kernel=copy_body_lambda_to_history,
        dim=(dims.N_w, dims.N_c),
        inputs=[data._body_lambda],
        outputs=[data.history._body_lambda_history[newton_iteration, :, :]],
        device=device,
    )


def perform_pca(
    trajectory_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs PCA on STANDARDIZED (Whitened) data.

    Returns: v1, v2, S, mean, std
    """
    # 1. Permute if 3D: (Steps, Worlds, Dofs) -> (Worlds, Steps, Dofs)
    # We want statistics over Steps (dim 1 in the permuted tensor)
    if trajectory_data.ndim == 3:
        data = trajectory_data.permute(1, 0, 2)
    else:
        data = trajectory_data.unsqueeze(0)

    # 2. Compute Statistics per World per DOF
    data_mean = data.mean(dim=1, keepdim=True)
    data_std = data.std(dim=1, keepdim=True)

    # 3. Robust Scaling (Whitening)
    # Instead of dividing by std (which explodes for stationary/constrained DOFs),
    # we clamp the denominator. If variation is below 'threshold', we don't scale it up.
    # This keeps noise as noise and prevents it from dominating the PCA.
    # Threshold heuristic: 1e-4 allows small valid motions but suppresses numerical noise.
    threshold = 1e-4
    scale = torch.maximum(data_std, torch.tensor(threshold, device=data.device))

    # 3. Standardize (Z-score normalization)
    # This brings q, u, lambda to the same ~unit scale
    data_normalized = (data - data_mean) / scale

    # 4. Perform SVD on Normalized Data
    try:
        k = min(2, data_normalized.shape[1], data_normalized.shape[2])
        _U, S, Vh = torch.linalg.svd(data_normalized, full_matrices=False)

        v1 = Vh[:, 0, :]
        v2 = Vh[:, 1, :] if k > 1 else torch.zeros_like(v1)
        S_out = S[:, :2]

    except torch.linalg.LinAlgError:
        print("SVD failed. Using random directions.")
        worlds, _, dofs = data.shape
        v1 = torch.randn(worlds, dofs, device=data.device)
        v1 /= torch.norm(v1, dim=-1, keepdim=True)
        v2 = torch.randn(worlds, dofs, device=data.device)
        v2 -= v1 * (v2 * v1).sum(dim=-1, keepdim=True)
        v2 /= torch.norm(v2, dim=-1, keepdim=True)
        S_out = torch.ones(worlds, 2, device=data.device)

    # Return mean/std so we can denormalize later
    return v1, v2, S_out, data_mean.squeeze(1), scale.squeeze(1)


def prepare_trajectory_data(
    q_history: torch.Tensor,  # Shape: (T, N_w, N_b, 7)
    u_history: torch.Tensor,  # Shape: (T, N_w, N_b, 6)
    lambda_history: torch.Tensor,  # Shape: (T, N_w, N_c)
    dt: float,
) -> torch.Tensor:
    """
    Flattens and scales q, u, and lambda into a unified 'velocity-level' feature vector.

    Scaling strategy:
    - q (Position/Rotation): Divided by dt -> Velocity / Angular Velocity units
    - u (Velocity): Kept as is -> Velocity units
    - lambda (Constraint Force): Multiplied by dt -> Impulse units
    """
    T = q_history.shape[0]

    # Flatten spatial/body dimensions
    q_flat = q_history.reshape(T, -1)
    u_flat = u_history.reshape(T, -1)
    lambda_flat = lambda_history.reshape(T, -1)

    # --- Apply Scaling ---

    # q: Position (m) and Quaternion (1). Divide by dt to get (m/s) and (1/s)
    q_scaled = q_flat / dt

    # u: Already in velocity units (rad/s, m/s). No scaling needed.
    u_scaled = u_flat

    # lambda: Force/Torque. Multiply by dt to get Impulse (N*s).
    # This brings it to momentum/velocity level (assuming mass ~ 1)
    lambda_scaled = lambda_flat * dt

    # Concatenate: [q', u', lambda']
    return torch.cat([q_scaled, u_scaled, lambda_scaled], dim=1)


def create_pca_grid_batch(
    x_center: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    S: torch.Tensor,
    mean: torch.Tensor,  # For Denormalization
    scale: torch.Tensor,  # Unused if scale=1, but kept for interface compatibility
    grid_res: int,
    plot_range_scale: float,
    dims: EngineDimensions,  # Need full dims to split q/u/lambda
    dt: float,  # NEW: Need dt to reverse the scaling
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = x_center.device

    # --- 1. Determine Plot Range ---
    if S.dim() == 2:
        max_bound_x = S[:, 0].max().item()
        max_bound_y = S[:, 1].max().item()
        if max_bound_y < 1e-6:
            max_bound_y = 0.1 * max_bound_x
    else:
        max_bound_x = S[0].item()
        max_bound_y = 0.1 * max_bound_x

    range_x = plot_range_scale * (max_bound_x if max_bound_x > 1e-6 else 1.0)
    range_y = plot_range_scale * (max_bound_y if max_bound_y > 1e-6 else 1.0)

    alphas = torch.linspace(-range_x, range_x, grid_res, device=device)
    betas = torch.linspace(-range_y, range_y, grid_res, device=device)
    alpha_grid, beta_grid = torch.meshgrid(alphas, betas, indexing="ij")

    # --- 2. Generate Grid in Feature Space (Velocity Level) ---
    # We perform PCA on standardized data (z-scores).
    # v1 and v2 are directions in that normalized space.
    # To step in physical space, we must re-apply the scale (std dev).
    # Grid units (alpha/beta) now correspond to "number of standard deviations".
    v1_phys = v1 * scale
    v2_phys = v2 * scale

    pca_batch_centered = (
        (x_center - mean).unsqueeze(0)
        + alpha_grid.flatten().view(-1, 1, 1) * v1_phys.unsqueeze(0)
        + beta_grid.flatten().view(-1, 1, 1) * v2_phys.unsqueeze(0)
    )

    # Add mean back to get absolute scaled values
    pca_batch_scaled = pca_batch_centered + mean.unsqueeze(0)

    # --- 3. Split [q', u', lambda'] ---
    num_bodies = dims.N_b
    dim_q = num_bodies * 7
    dim_u = num_bodies * 6

    pca_q_scaled = pca_batch_scaled[:, :, :dim_q]
    pca_u_scaled = pca_batch_scaled[:, :, dim_q : dim_q + dim_u]
    pca_lambda_scaled = pca_batch_scaled[:, :, dim_q + dim_u :]

    # --- 4. Un-Scale back to Physics Units ---

    # q = q' * dt
    pca_q_flat = pca_q_scaled * dt

    # u = u'
    pca_u_flat = pca_u_scaled

    # lambda = lambda' / dt
    pca_lambda_flat = pca_lambda_scaled / dt

    # --- 5. Normalize Quaternions in Q ---
    pca_q_reshaped = pca_q_flat.view(-1, dims.N_w, dims.N_b, 7)
    pos = pca_q_reshaped[..., :3]
    rot = pca_q_reshaped[..., 3:]
    rot = torch.nn.functional.normalize(rot, p=2, dim=-1)

    pca_q_final = torch.cat([pos, rot], dim=-1)

    return (
        pca_q_final.reshape(-1, dims.N_w, dim_q),
        pca_u_flat,
        pca_lambda_flat,
        alphas,
        betas,
        alpha_grid,
        beta_grid,
    )


@wp.kernel
def copy_spatial_to_flat_kernel(
    src_spatial: wp.array(dtype=wp.spatial_vector, ndim=3),
    dst_flat: wp.array(dtype=float, ndim=3),
):
    g, w, b = wp.tid()
    # dst_flat has shape (G, W, Nu+Nc)
    # We write to the first Nu elements.
    # Base index for body b is b*6.

    val = src_spatial[g, w, b]
    base = b * 6

    dst_flat[g, w, base + 0] = val[0]
    dst_flat[g, w, base + 1] = val[1]
    dst_flat[g, w, base + 2] = val[2]
    dst_flat[g, w, base + 3] = val[3]
    dst_flat[g, w, base + 4] = val[4]
    dst_flat[g, w, base + 5] = val[5]


def compute_pca_batch_h_norm(
    model: AxionModel,
    data: EngineData,
    config: EngineConfig,
    dims: EngineDimensions,
):
    device = data.device
    data.history.pca_batch_h.full.zero_()

    B = data.history.pca_batch_body_u.shape[0]  # Grid Size

    # Launch kernels over (GridSize, N_w, ...)
    # All kernels support batch dimensions naturally
    wp.launch(
        kernel=batch_unconstrained_dynamics_kernel,
        dim=(B, dims.N_w, dims.N_b),
        inputs=[
            data.history.pca_batch_body_q,
            data.history.pca_batch_body_u,
            data.body_u_prev,
            data.body_f,
            model.body_mass,
            model.body_inertia,
            data.dt,
            data.g_accel,
        ],
        outputs=[data.history.pca_batch_h.d_spatial],
        device=device,
    )

    if config.joint_constraint_level == "pos":
        # wp.launch(
        #     kernel=batch_positional_joint_residual_kernel,
        #     dim=(B, dims.N_w, dims.N_j),
        #     inputs=[
        #         data.history.pca_batch_body_u,
        #         data.history.pca_batch_body_lambda.j,
        #         data.joint_constraint_data,
        #         data.dt,
        #         config.joint_stabilization_factor,
        #         config.joint_compliance,
        #     ],
        #     outputs=[
        #         data.history.pca_batch_h.d_spatial,
        #         data.history.pca_batch_h.c.j,
        #     ],
        #     device=device,
        # )
        wp.launch(
            kernel=batch_positional_joint_residual_kernel,
            dim=(B, dims.N_w, dims.joint_count),
            inputs=[
                data.history.pca_batch_body_q,
                data.history.pca_batch_body_lambda.j,
                model.body_com,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_qd_start,
                model.joint_enabled,
                data.joint_constraint_offsets,
                data.dt,
                config.joint_compliance,
            ],
            outputs=[
                data.history.pca_batch_h.d_spatial,
                data.history.pca_batch_h.c.j,
            ],
            device=device,
        )
    elif config.joint_constraint_level == "vel":
        wp.launch(
            kernel=batch_velocity_joint_residual_kernel,
            dim=(B, dims.N_w, dims.N_j),
            inputs=[
                data.history.pca_batch_body_u,
                data.history.pca_batch_body_lambda.j,
                data.joint_constraint_data,
                data.dt,
                config.joint_stabilization_factor,
                config.joint_compliance,
            ],
            outputs=[
                data.history.pca_batch_h.d_spatial,
                data.history.pca_batch_h.c.j,
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
                data.history.pca_batch_body_q,
                data.history.pca_batch_body_u,
                data.body_u_prev,
                data.history.pca_batch_body_lambda.n,
                data.contact_interaction,
                model.body_inv_mass,
                model.body_inv_inertia,
                data.dt,
                config.contact_fb_alpha,
                config.contact_fb_beta,
                config.contact_compliance,
            ],
            outputs=[
                data.history.pca_batch_h.d_spatial,
                data.history.pca_batch_h.c.n,
            ],
            device=device,
        )
    elif config.contact_constraint_level == "vel":
        wp.launch(
            kernel=batch_velocity_contact_residual_kernel,
            dim=(B, dims.N_w, dims.N_n),
            inputs=[
                data.history.pca_batch_body_u,
                data.body_u_prev,
                data.history.pca_batch_body_lambda.n,
                data.contact_interaction,
                data.dt,
                config.contact_stabilization_factor,
                config.contact_fb_alpha,
                config.contact_fb_beta,
                config.contact_compliance,
            ],
            outputs=[
                data.history.pca_batch_h.d_spatial,
                data.history.pca_batch_h.c.n,
            ],
            device=device,
        )
    else:
        raise ValueError("Contact constraint level can be only 'pos' or 'vel'.")

    wp.launch(
        kernel=batch_friction_residual_kernel,
        dim=(B, dims.N_w, dims.N_n),
        inputs=[
            data.history.pca_batch_body_q,
            data.history.pca_batch_body_u,
            data.history.pca_batch_body_lambda.f,
            data.body_lambda_prev.f,
            data.body_lambda_prev.n,
            data.s_n_prev,
            data.contact_interaction,
            model.body_inv_mass,
            model.body_inv_inertia,
            data.dt,
            config.friction_fb_alpha,
            config.friction_fb_beta,
            config.friction_compliance,
        ],
        outputs=[
            data.history.pca_batch_h.d_spatial,
            data.history.pca_batch_h.c.f,
        ],
        device=device,
    )

    # Compute Norm
    # data.pca_batch_h.full is (GridSize, N_w, Dofs)
    torch_h = wp.to_torch(data.history.pca_batch_h.full)
    torch_h_norm = torch.norm(torch_h, dim=-1)  # -> (GridSize, N_w)

    h_norm = wp.from_torch(torch_h_norm)
    wp.copy(data.history.pca_batch_h_norm, h_norm.contiguous())

    return torch_h_norm
