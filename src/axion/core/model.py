import numpy as np
import warp as wp
from newton import Model


@wp.kernel
def batch_joint_q_start_kernel(
    joint_count_per_world: wp.int32,
    joint_coord_count_per_world: wp.int32,
    joint_q_start: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_joint_q_start: wp.array(dtype=wp.int32, ndim=2),
):
    joint_idx = wp.tid()
    if joint_idx >= joint_q_start.shape[0] - 1:
        return

    world_idx = joint_idx // joint_count_per_world
    slot = joint_idx % joint_count_per_world

    if joint_coord_count_per_world > 0:
        batched_joint_q_start[world_idx, slot] = (
            joint_q_start[joint_idx] % joint_coord_count_per_world
        )
    else:
        batched_joint_q_start[world_idx, slot] = 0


@wp.kernel
def batch_joint_qd_start_kernel(
    joint_count_per_world: wp.int32,
    joint_dof_count_per_world: wp.int32,
    joint_qd_start: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
):
    joint_idx = wp.tid()
    if joint_idx >= joint_qd_start.shape[0] - 1:
        return

    world_idx = joint_idx // joint_count_per_world
    slot = joint_idx % joint_count_per_world

    if joint_dof_count_per_world > 1:
        batched_joint_qd_start[world_idx, slot] = joint_qd_start[joint_idx] % (
            joint_dof_count_per_world - 1
        )
    else:
        batched_joint_qd_start[world_idx, slot] = 0


@wp.kernel
def batch_joint_bodies_kernel(
    joint_count_per_world: wp.int32,
    body_count_per_world: wp.int32,
    joint_parent: wp.array(dtype=wp.int32),
    joint_child: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_joint_parent: wp.array(dtype=wp.int32, ndim=2),
    batched_joint_child: wp.array(dtype=wp.int32, ndim=2),
):
    joint_idx = wp.tid()

    if joint_idx >= joint_parent.shape[0]:
        return

    world_idx = joint_idx // joint_count_per_world
    slot = joint_idx % joint_count_per_world

    parent_idx = joint_parent[joint_idx]
    if parent_idx >= 0:
        batched_joint_parent[world_idx, slot] = parent_idx % body_count_per_world
    else:
        batched_joint_parent[world_idx, slot] = parent_idx

    child_idx = joint_child[joint_idx]
    if child_idx >= 0:
        batched_joint_child[world_idx, slot] = child_idx % body_count_per_world
    else:
        batched_joint_child[world_idx, slot] = child_idx


@wp.kernel
def batch_shape_body_kernel(
    shape_count_per_world: wp.int32,
    body_count_per_world: wp.int32,
    shape_body: wp.array(dtype=wp.int32),
    # OUTPUT
    batched_shape_body: wp.array(dtype=wp.int32, ndim=2),
):
    shape_idx = wp.tid()

    if shape_idx >= shape_body.shape[0]:
        return

    world_idx = shape_idx // shape_count_per_world
    slot = shape_idx % shape_count_per_world

    body_idx = shape_body[shape_idx]
    if body_idx >= 0:
        batched_shape_body[world_idx, slot] = body_idx % body_count_per_world
    else:
        batched_shape_body[world_idx, slot] = body_idx


def compute_joint_constraint_offsets(joint_types: wp.array):
    """
    joint_types: numpy array of shape (num_worlds, num_joints)
    """

    constraint_count_map = np.array(
        [
            5,  # PRISMATIC = 0
            5,  # REVOLUTE  = 1
            3,  # BALL      = 2
            6,  # FIXED     = 3
            0,  # FREE      = 4
            1,  # DISTANCE  = 5
            6,  # D6        = 6
            0,  # CABLE     = 7
        ],
        dtype=np.int32,
    )

    joint_types_np = joint_types.numpy()  # (num_worlds, num_joints)
    # Map joint types → constraint counts
    constraint_counts = constraint_count_map[joint_types_np]  # (num_worlds, num_joints)

    # Total constraints for each batch
    total_constraints = constraint_counts.sum(axis=1)  # (num_worlds,)

    # Compute offsets per batch
    # For each batch: offsets[i, :] = cumsum(counts[i, :]) - counts[i, 0]
    constraint_offsets = np.zeros_like(constraint_counts)  # (num_worlds, num_joints)
    constraint_offsets[:, 1:] = np.cumsum(constraint_counts[:, :-1], axis=1)

    # Convert to wp.array (must flatten or provide device explicitly)
    constraint_offsets_wp = wp.array(
        constraint_offsets,
        dtype=wp.int32,
        device=joint_types.device,
    )

    return constraint_offsets_wp, total_constraints[0]


@wp.kernel
def count_control_constraints_kernel(
    joint_type: wp.array(dtype=wp.int32, ndim=2),
    joint_dof_mode: wp.array(dtype=wp.int32, ndim=2),
    joint_qd_start: wp.array(dtype=wp.int32, ndim=2),
    counts: wp.array(dtype=wp.int32, ndim=2),
):
    world_idx, joint_idx = wp.tid()
    j_type = joint_type[world_idx, joint_idx]
    count = 0
    if j_type == 1 or j_type == 0:
        qd_start = joint_qd_start[world_idx, joint_idx]
        mode = joint_dof_mode[world_idx, qd_start]
        if mode != 0:
            count = 1
    counts[world_idx, joint_idx] = count


def compute_control_constraint_offsets(
    joint_type: wp.array,
    joint_dof_mode: wp.array,
    joint_qd_start: wp.array,
):
    num_worlds = joint_type.shape[0]
    num_joints = joint_type.shape[1]
    counts = wp.zeros((num_worlds, num_joints), dtype=wp.int32, device=joint_type.device)

    wp.launch(
        kernel=count_control_constraints_kernel,
        dim=(num_worlds, num_joints),
        inputs=[joint_type, joint_dof_mode, joint_qd_start],
        outputs=[counts],
        device=joint_type.device,
    )

    counts_np = counts.numpy()

    offsets_np = np.zeros_like(counts_np)
    row_counts = counts_np[0]
    row_offsets = np.cumsum(np.concatenate(([0], row_counts[:-1])))
    total = np.sum(row_counts)
    offsets_np[:] = row_offsets
    offsets = wp.from_numpy(offsets_np, dtype=wp.int32, device=joint_type.device)

    return offsets, int(total)


class AxionModel:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.device = model.device
        self.num_worlds = model.num_worlds

        self.g_accel = model.gravity

        self.body_count = model.body_count // model.num_worlds
        self.shape_count = model.shape_count // model.num_worlds
        self.joint_count = model.joint_count // model.num_worlds
        self.joint_dof_count = model.joint_dof_count // model.num_worlds
        self.joint_coord_count = model.joint_coord_count // model.num_worlds

        with wp.ScopedDevice(self.device):
            self.body_com = model.body_com.reshape((model.num_worlds, -1))
            self.body_inertia = model.body_inertia.reshape((model.num_worlds, -1))
            self.body_inv_inertia = model.body_inv_inertia.reshape((model.num_worlds, -1))
            self.body_mass = model.body_mass.reshape((model.num_worlds, -1))
            self.body_inv_mass = model.body_inv_mass.reshape((model.num_worlds, -1))

            self.joint_X_c = model.joint_X_c.reshape((model.num_worlds, -1))
            self.joint_X_p = model.joint_X_p.reshape((model.num_worlds, -1))
            self.joint_axis = model.joint_axis.reshape((model.num_worlds, -1))
            self.joint_dof_dim = model.joint_dof_dim.reshape((model.num_worlds, -1))
            self.joint_dof_mode = model.joint_dof_mode.reshape((model.num_worlds, -1))
            self.joint_enabled = model.joint_enabled.reshape((model.num_worlds, -1))
            self.joint_target_ke = model.joint_target_ke.reshape((model.num_worlds, -1))
            self.joint_target_kd = model.joint_target_kd.reshape((model.num_worlds, -1))
            self.joint_compliance = model.joint_compliance.reshape((model.num_worlds, -1))
            self.joint_type = model.joint_type.reshape((model.num_worlds, -1))
            self.joint_q_start = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)
            self.joint_qd_start = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)
            self.joint_parent = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)
            self.joint_child = wp.zeros((model.num_worlds, self.joint_count), dtype=wp.int32)

        wp.launch(
            kernel=batch_joint_q_start_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.joint_count,
                self.joint_coord_count,
                model.joint_q_start,
            ],
            outputs=[
                self.joint_q_start,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=batch_joint_qd_start_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.joint_count,
                self.joint_coord_count,
                model.joint_qd_start,
            ],
            outputs=[
                self.joint_qd_start,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=batch_joint_bodies_kernel,
            dim=self.model.joint_count,
            inputs=[
                self.joint_count,
                self.body_count,
                model.joint_parent,
                model.joint_child,
            ],
            outputs=[
                self.joint_parent,
                self.joint_child,
            ],
            device=self.device,
        )

        with wp.ScopedDevice(self.device):
            self.shape_body = wp.zeros((model.num_worlds, self.shape_count), dtype=wp.int32)
            self.shape_thickness = model.shape_thickness.reshape((model.num_worlds, -1))
            self.shape_material_mu = model.shape_material_mu.reshape((model.num_worlds, -1))
            self.shape_material_restitution = model.shape_material_restitution.reshape(
                (model.num_worlds, -1)
            )

        wp.launch(
            kernel=batch_shape_body_kernel,
            dim=self.model.shape_count,
            inputs=[
                self.shape_count,
                self.body_count,
                model.shape_body,
            ],
            outputs=[
                self.shape_body,
            ],
            device=self.device,
        )

        self.joint_constraint_offsets, self.num_joint_constraints = (
            compute_joint_constraint_offsets(
                self.joint_type,
            )
        )

        self.control_constraint_offsets, self.num_control_constraints = (
            compute_control_constraint_offsets(
                self.joint_type,
                self.joint_dof_mode,
                self.joint_qd_start,
            )
        )
