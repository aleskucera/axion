import torch

def _normalize_vec(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def _batch_axis_angle_to_rotmat(axis_b3: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues formula for a batch of axis-angle rotations.
    axis_b3: (B, 3), angle_b: (B,)
    returns: (B, 3, 3)
    """
    axis_b3 = _normalize_vec(axis_b3)
    x, y, z = axis_b3[:, 0], axis_b3[:, 1], axis_b3[:, 2]
    c = torch.cos(angle_b)
    s = torch.sin(angle_b)
    one_c = 1.0 - c

    r00 = c + x * x * one_c
    r01 = x * y * one_c - z * s
    r02 = x * z * one_c + y * s
    r10 = y * x * one_c + z * s
    r11 = c + y * y * one_c
    r12 = y * z * one_c - x * s
    r20 = z * x * one_c - y * s
    r21 = z * y * one_c + x * s
    r22 = c + z * z * one_c

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=1,
    )


def _batch_quat_xyzw_to_rotmat(quat_b4: torch.Tensor) -> torch.Tensor:
    """Convert batched xyzw quaternions to rotation matrices."""
    q = _normalize_vec(quat_b4)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=1,
    )


def pendulum_revolute_minimal_to_maximal_velocities(
    q_b2: torch.Tensor,
    qd_b2: torch.Tensor,
    joint_axis_parent_23: torch.Tensor,
    joint_x_p_27: torch.Tensor,
    joint_x_c_27: torch.Tensor,
    body_com_23: torch.Tensor,
    body_count: int = 2,
) -> torch.Tensor:
    """
    Differentiable Torch kinematics map from minimal pendulum (q, qd) to maximal body velocities.

    This mirrors the Newton articulation FK velocity recursion for a 2-link revolute chain:
      - world parent anchor velocity propagation
      - joint angular velocity injection
      - child COM velocity from child-anchor offset

    Args:
        q_b2: (B, 2) generalized joint positions.
        qd_b2: (B, 2) generalized joint velocities.
        joint_axis_parent_23: (2, 3) revolute axes in each joint's parent frame.
        joint_x_p_27: (2, 7) parent-anchor transforms from joint_X_p (xyz + xyzw quat).
        joint_x_c_27: (2, 7) child-anchor transforms from joint_X_c (xyz + xyzw quat).
        body_com_23: (2, 3) body COM offsets in body-local frame.
        body_count: number of articulated bodies per world (expected 2 for double pendulum).

    Returns:
        Flattened maximal spatial body velocities, shape (B, body_count * 6),
        where each 6-vector is [v_xyz, w_xyz] in world coordinates.
    """
    if q_b2.ndim != 2 or qd_b2.ndim != 2:
        raise RuntimeError(
            f"Expected q/qd with shape (B,2), got q={tuple(q_b2.shape)}, qd={tuple(qd_b2.shape)}."
        )
    if q_b2.shape != qd_b2.shape:
        raise RuntimeError(
            f"q and qd shape mismatch: q={tuple(q_b2.shape)}, qd={tuple(qd_b2.shape)}."
        )
    if q_b2.shape[1] != 2:
        raise RuntimeError(f"Expected pendulum minimal dimension 2, got {q_b2.shape[1]}.")
    if body_count != 2:
        raise RuntimeError(f"This function currently supports body_count=2 only, got {body_count}.")
    if joint_axis_parent_23.shape != (2, 3):
        raise RuntimeError(
            f"Expected joint_axis_parent_23 shape (2,3), got {tuple(joint_axis_parent_23.shape)}."
        )
    if joint_x_p_27.shape != (2, 7):
        raise RuntimeError(
            f"Expected joint_x_p_27 shape (2,7), got {tuple(joint_x_p_27.shape)}."
        )
    if joint_x_c_27.shape != (2, 7):
        raise RuntimeError(
            f"Expected joint_x_c_27 shape (2,7), got {tuple(joint_x_c_27.shape)}."
        )
    if body_com_23.shape != (2, 3):
        raise RuntimeError(
            f"Expected body_com_23 shape (2,3), got {tuple(body_com_23.shape)}."
        )

    batch = q_b2.shape[0]
    device = q_b2.device
    dtype = q_b2.dtype

    axes = joint_axis_parent_23.to(device=device, dtype=dtype)
    x_p_pos = joint_x_p_27[:, :3].to(device=device, dtype=dtype)
    x_p_quat = joint_x_p_27[:, 3:].to(device=device, dtype=dtype)
    x_c_pos = joint_x_c_27[:, :3].to(device=device, dtype=dtype)
    x_c_quat = joint_x_c_27[:, 3:].to(device=device, dtype=dtype)
    com = body_com_23.to(device=device, dtype=dtype)

    # Parent world state (joint 0 parent is static world frame).
    r_parent = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch, 3, 3)
    p_parent = torch.zeros(batch, 3, device=device, dtype=dtype)
    v_parent = torch.zeros(batch, 3, device=device, dtype=dtype)
    w_parent = torch.zeros(batch, 3, device=device, dtype=dtype)

    body_linear = []
    body_angular = []

    for joint_idx in range(2):
        r_pj = _batch_quat_xyzw_to_rotmat(x_p_quat[joint_idx].unsqueeze(0).expand(batch, 4))
        r_cj = _batch_quat_xyzw_to_rotmat(x_c_quat[joint_idx].unsqueeze(0).expand(batch, 4))

        # Parent anchor transform in world: X_wpj = X_wp * X_pj
        r_wpj = torch.bmm(r_parent, r_pj)
        p_wpj = p_parent + torch.bmm(
            r_parent, x_p_pos[joint_idx].view(1, 3, 1).expand(batch, 3, 1)
        ).squeeze(-1)

        axis_parent = axes[joint_idx].unsqueeze(0).expand(batch, 3)
        axis_wpj = torch.bmm(r_wpj, axis_parent.unsqueeze(-1)).squeeze(-1)

        q_i = q_b2[:, joint_idx]
        qd_i = qd_b2[:, joint_idx]

        r_joint = _batch_axis_angle_to_rotmat(axis_parent, q_i)
        # Newton: v_wpj from parent body COM velocity and lever arm r_p.
        com_parent_world = p_parent + torch.bmm(
            r_parent, com[joint_idx].view(1, 3, 1).expand(batch, 3, 1)
        ).squeeze(-1)
        r_p = p_wpj - com_parent_world
        v_wpj_linear = v_parent + torch.cross(w_parent, r_p, dim=-1)
        v_wpj_angular = w_parent

        # Newton: transform joint twist through X_wpj.
        linear_vel = torch.zeros(batch, 3, device=device, dtype=dtype)
        angular_vel = axis_wpj * qd_i.unsqueeze(-1)

        # v_wc = v_wpj + transformed v_j
        v_child = v_wpj_linear + linear_vel
        w_child = v_wpj_angular + angular_vel

        # Newton pose path:
        # X_wcj = X_wpj * X_j, X_wc = X_wcj * inverse(X_cj)
        r_wcj = torch.bmm(r_wpj, r_joint)
        r_child = torch.bmm(r_wcj, r_cj.transpose(1, 2))
        p_wcj = p_wpj  # revolute X_j has zero translation
        p_child = p_wcj - torch.bmm(
            r_child, x_c_pos[joint_idx].view(1, 3, 1).expand(batch, 3, 1)
        ).squeeze(-1)

        body_linear.append(v_child)
        body_angular.append(w_child)

        # Propagate for next joint in chain.
        r_parent = r_child
        p_parent = p_child
        v_parent = v_child
        w_parent = w_child

    body_qd_b26 = torch.cat(
        [
            torch.cat([body_linear[0], body_angular[0]], dim=-1),
            torch.cat([body_linear[1], body_angular[1]], dim=-1),
        ],
        dim=-1,
    )
    return body_qd_b26

################################################################################
# LOCAL TESTING:
################################################################################

if __name__ == "__main__":
    import argparse
    import os
    import sys

    import newton
    import warp as wp

    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_this_dir, "..", "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from examples.double_pendulum.pendulum_articulation_definition import build_pendulum_model

    parser = argparse.ArgumentParser(
        description="Local test for pendulum_revolute_minimal_to_maximal_velocities."
    )
    parser.add_argument("--num-worlds", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    wp_device = wp.get_device(args.device)
    torch_device = wp.device_to_torch(wp_device)

    model = build_pendulum_model(
        num_worlds=args.num_worlds,
        device=wp_device,
        requires_grad=False,
    )
    state = model.state()

    joints_per_world = model.joint_count // model.world_count
    bodies_per_world = model.body_count // model.world_count
    if joints_per_world != 2:
        raise RuntimeError(
            f"Expected 2 joints per world for pendulum test, got {joints_per_world}."
        )
    if bodies_per_world != 2:
        raise RuntimeError(
            f"Expected 2 bodies per world for pendulum test, got {bodies_per_world}."
        )

    # Sample random generalized coordinates.
    q_b2 = (2.0 * torch.rand(args.num_worlds, 2, device=torch_device) - 1.0) * torch.pi
    qd_b2 = (2.0 * torch.rand(args.num_worlds, 2, device=torch_device) - 1.0) * 4.0

    wp.copy(state.joint_q, wp.from_torch(q_b2.reshape(-1).contiguous(), dtype=wp.float32))
    wp.copy(state.joint_qd, wp.from_torch(qd_b2.reshape(-1).contiguous(), dtype=wp.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # Reference maximal body velocities from Newton FK.
    body_qd_ref = wp.to_torch(state.body_qd).to(torch_device)
    body_qd_ref = body_qd_ref.reshape(args.num_worlds, bodies_per_world, 6).reshape(args.num_worlds, -1)

    # Extract constants from model (world-0 slice, replicated worlds).
    joint_axis = wp.to_torch(model.joint_axis).to(torch_device).reshape(args.num_worlds, joints_per_world, 3)[0]
    joint_x_p = wp.to_torch(model.joint_X_p).to(torch_device).reshape(args.num_worlds, joints_per_world, 7)[0]
    joint_x_c = wp.to_torch(model.joint_X_c).to(torch_device).reshape(args.num_worlds, joints_per_world, 7)[0]
    body_com = wp.to_torch(model.body_com).to(torch_device).reshape(args.num_worlds, bodies_per_world, 3)[0]

    body_qd_pred = pendulum_revolute_minimal_to_maximal_velocities(
        q_b2=q_b2,
        qd_b2=qd_b2,
        joint_axis_parent_23=joint_axis,
        joint_x_p_27=joint_x_p,
        joint_x_c_27=joint_x_c,
        body_com_23=body_com,
        body_count=bodies_per_world,
    )

    abs_err = (body_qd_pred - body_qd_ref).abs()
    max_err = abs_err.max().item()
    mean_err = abs_err.mean().item()
    per_body_max = abs_err.reshape(args.num_worlds, bodies_per_world, 6).amax(dim=0)
    print(f"[FK test] num_worlds={args.num_worlds}, device={args.device}")
    print(f"[FK test] max_abs_err={max_err:.6e}, mean_abs_err={mean_err:.6e}, tol={args.tol:.1e}")
    print("[FK test] per_body_max_abs_err body0[vx..wz], body1[vx..wz]:")
    print(per_body_max)

    # Simple differentiability check.
    q_var = q_b2.clone().detach().requires_grad_(True)
    qd_var = qd_b2.clone().detach().requires_grad_(True)
    out = pendulum_revolute_minimal_to_maximal_velocities(
        q_b2=q_var,
        qd_b2=qd_var,
        joint_axis_parent_23=joint_axis,
        joint_x_p_27=joint_x_p,
        joint_x_c_27=joint_x_c,
        body_com_23=body_com,
        body_count=bodies_per_world,
    )
    loss = (out ** 2).mean()
    grad_q, grad_qd = torch.autograd.grad(loss, [q_var, qd_var], retain_graph=False, allow_unused=False)
    print(
        "[FK grad] grad_q_l1={:.6e}, grad_qd_l1={:.6e}".format(
            grad_q.abs().sum().item(),
            grad_qd.abs().sum().item(),
        )
    )

    if max_err > args.tol:
        raise SystemExit(1)