import numpy as np
import warp as wp

wp.init()

NUM_WORLDS = 200
NUM_JOINT_CONSTRAINTS_PER_WORLD = 3
NUM_BODIES_PER_WORLD = 4
MAX_GROUND_CONTACTS_PER_BODY = 16
MAX_CONTACTS_BETWEEN_BODIES_PER_WORLD = 8

NUM_JOINT_CONSTRAINTS = NUM_WORLDS * NUM_JOINT_CONSTRAINTS_PER_WORLD
NUM_BODIES = NUM_WORLDS * NUM_BODIES_PER_WORLD


@wp.kernel
def kernel_J_matvec(
    x: wp.array(dtype=wp.spatial_vector, ndim=2),
    y: wp.array(dtype=wp.float32, ndim=2),
    J_values: wp.array(dtype=wp.spatial_vector, ndim=3),
    constraint_body_idx: wp.array(dtype=wp.int32, ndim=3),
    alpha: float,
    beta: float,
    # Output array
    z: wp.array(dtype=wp.float32, ndim=2),
):
    """
    z = alpha * (J @ x) + beta * y
    """
    world_idx, constraint_idx = wp.tid()

    # The result of A @ x.
    # Default is 0.0 (implied for inactive constraints)
    body_1 = constraint_body_idx[world_idx, constraint_idx, 0]
    body_2 = constraint_body_idx[world_idx, constraint_idx, 1]
    J_1 = J_values[world_idx, constraint_idx, 0]
    J_2 = J_values[world_idx, constraint_idx, 1]

    j_x = 0.0
    if body_1 >= 0:
        j_x += wp.dot(J_1, x[world_idx, body_1])
    if body_2 >= 0:
        j_x += wp.dot(J_2, x[world_idx, body_2])

    # Final composition.
    if beta == 0.0:
        z[world_idx, constraint_idx] = alpha * j_x
    else:
        y_val = y[world_idx, constraint_idx]
        z[world_idx, constraint_idx] = alpha * j_x + beta * y_val


class OldOperator:
    def __init__(
        self,
        device: wp.context.Device,
        num_worlds: int,
        num_bodies: int,
        num_joint_constraints: int,
        max_ground_contacts_per_body: int,
        max_contacts_between_bodies_per_world: int,
    ):
        self.device = device

        self.num_worlds = num_worlds
        self.num_bodies = num_bodies
        self.num_joint_constraints = num_joint_constraints
        self.max_ground_contacts_per_body = max_ground_contacts_per_body
        self.max_contacts_between_bodies_per_world = max_contacts_between_bodies_per_world

        self.shape = (
            self.num_worlds,
            self.num_joint_constraints // self.num_worlds
            + 3 * self.max_ground_contacts_per_body * self.num_bodies // self.num_worlds
            + 3 * self.max_contacts_between_bodies_per_world,
            self.num_bodies // self.num_worlds,
        )

        self.J_values = None
        self.constraint_body_idx = None

        self._init_sparse_matrix()

    def _init_sparse_matrix(self):
        num_worlds = self.shape[0]
        constraints_per_world = self.shape[1]
        bodies_per_world = self.shape[2]

        body_indices = []
        for w in range(num_worlds):
            body_indices_per_world = []

            for c in range(constraints_per_world):
                rand_body_indices = np.random.choice(bodies_per_world, 2, replace=False)
                body_indices_per_world.append(rand_body_indices.tolist())

            body_indices.append(body_indices_per_world)

        self.constraint_body_idx = wp.array(body_indices, dtype=wp.int32)

        J_values_np = np.random.rand(num_worlds, constraints_per_world, 2, 6)
        self.J_values = wp.from_numpy(J_values_np, dtype=wp.spatial_vector)

    def matvec(self, x, y, z, alpha, beta):
        wp.launch(
            kernel=kernel_J_matvec,
            dim=(self.shape[0], self.shape[1]),
            inputs=[
                x,
                y,
                self.J_values,
                self.constraint_body_idx,
                alpha,
                beta,
            ],
            outputs=[z],
        )
