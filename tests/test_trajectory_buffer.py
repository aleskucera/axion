import unittest

import numpy as np
import warp as wp
from axion.core.engine_config import AxionEngineConfig
from axion.core.engine_data import EngineData
from axion.core.engine_dims import EngineDimensions
from axion.simulation.trajectory_buffer import TrajectoryBuffer


class TestTrajectoryBuffer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        wp.init()
        cls.device = wp.get_device("cuda:0" if wp.get_device("cuda").is_cuda else "cpu")

    def create_dummy_dims(self):
        """Creates small dimensions sufficient for testing."""
        return EngineDimensions(
            num_worlds=2,  # Test batching
            body_count=3,  # Test body arrays
            contact_count=5,  # Test contact capacity
            joint_count=2,
            joint_dof_count=2,
            linesearch_step_count=1,
            joint_constraint_count=2,
            control_constraint_count=1,  # Added control to ensure N_c > joint_constraints
            # Implicitly sets N_n (max contacts) if your EngineDimensions calculates it,
            # otherwise ensure N_n is set. Assuming standard Axion dims here.
        )

    def create_dummy_data(self, dims):
        """
        Creates an EngineData instance.
        """
        # We need to minimally initialize EngineData.
        # Let's trust EngineData.create works with zeros/dummy offsets
        zeros_int = wp.zeros(dims.joint_count, dtype=int, device=self.device)

        data = EngineData.create(
            dims=dims,
            config=AxionEngineConfig(),
            joint_constraint_offsets=zeros_int,
            control_constraint_offsets=zeros_int,
            dof_count=dims.joint_dof_count,
            device=self.device,
            allocate_history=False,
        )
        return data

    def test_round_trip_correctness(self):
        """
        Saves random data to the buffer and ensures it loads back exactly.
        Validates the unified lambda storage logic.
        """
        dims = self.create_dummy_dims()
        num_steps = 5
        buffer = TrajectoryBuffer(dims, num_steps, self.device)
        data = self.create_dummy_data(dims)

        # 1. Generate unique random patterns for each step
        ground_truth_history = []

        for t in range(num_steps):
            # Dynamics
            body_q_np = np.random.rand(dims.num_worlds, dims.body_count, 7).astype(np.float32)
            body_q_prev_np = np.random.rand(dims.num_worlds, dims.body_count, 7).astype(np.float32)
            body_u_np = np.random.rand(dims.num_worlds, dims.body_count, 6).astype(np.float32)
            body_u_prev_np = np.random.rand(dims.num_worlds, dims.body_count, 6).astype(np.float32)
            body_f_np = np.random.rand(dims.num_worlds, dims.body_count, 6).astype(np.float32)

            # Inputs
            joint_target_pos_np = np.random.rand(dims.num_worlds, dims.joint_dof_count).astype(np.float32)
            joint_target_vel_np = np.random.rand(dims.num_worlds, dims.joint_dof_count).astype(np.float32)

            # Constraints
            body_lambda_np = np.random.rand(dims.num_worlds, dims.N_c).astype(np.float32)
            body_lambda_prev_np = np.random.rand(dims.num_worlds, dims.N_c).astype(np.float32)
            constraint_active_mask_np = np.random.rand(dims.num_worlds, dims.N_c).astype(np.float32)
            constraint_body_idx_np = np.random.randint(0, dims.body_count, size=(dims.num_worlds, dims.N_c, 2)).astype(np.int32)

            # Contacts
            contact_body_a_np = np.random.randint(0, dims.body_count, size=(dims.num_worlds, dims.contact_count)).astype(np.int32)
            contact_body_b_np = np.random.randint(0, dims.body_count, size=(dims.num_worlds, dims.contact_count)).astype(np.int32)
            contact_point_a_np = np.random.rand(dims.num_worlds, dims.contact_count, 3).astype(np.float32)
            contact_point_b_np = np.random.rand(dims.num_worlds, dims.contact_count, 3).astype(np.float32)
            contact_basis_n_a_np = np.random.rand(dims.num_worlds, dims.contact_count, 6).astype(np.float32)
            contact_dist_np = np.random.rand(dims.num_worlds, dims.contact_count).astype(np.float32)
            contact_friction_coeff_np = np.random.rand(dims.num_worlds, dims.contact_count).astype(np.float32)

            with wp.ScopedDevice(self.device):
                # Dynamics
                wp.copy(data.body_q, wp.from_numpy(body_q_np, dtype=wp.transform))
                wp.copy(data.body_q_prev, wp.from_numpy(body_q_prev_np, dtype=wp.transform))
                wp.copy(data.body_u, wp.from_numpy(body_u_np, dtype=wp.spatial_vector))
                wp.copy(data.body_u_prev, wp.from_numpy(body_u_prev_np, dtype=wp.spatial_vector))
                wp.copy(data.body_f, wp.from_numpy(body_f_np, dtype=wp.spatial_vector))

                # Inputs
                wp.copy(data.joint_target_pos, wp.from_numpy(joint_target_pos_np, dtype=wp.float32))
                wp.copy(data.joint_target_vel, wp.from_numpy(joint_target_vel_np, dtype=wp.float32))

                # Constraints
                wp.copy(data._body_lambda, wp.from_numpy(body_lambda_np, dtype=wp.float32))
                wp.copy(data._body_lambda_prev, wp.from_numpy(body_lambda_prev_np, dtype=wp.float32))
                wp.copy(data._constraint_active_mask, wp.from_numpy(constraint_active_mask_np, dtype=wp.float32))
                wp.copy(data._constraint_body_idx, wp.from_numpy(constraint_body_idx_np, dtype=wp.int32))

                # Contacts
                wp.copy(data.contact_body_a, wp.from_numpy(contact_body_a_np, dtype=wp.int32))
                wp.copy(data.contact_body_b, wp.from_numpy(contact_body_b_np, dtype=wp.int32))
                wp.copy(data.contact_point_a, wp.from_numpy(contact_point_a_np, dtype=wp.vec3))
                wp.copy(data.contact_point_b, wp.from_numpy(contact_point_b_np, dtype=wp.vec3))
                wp.copy(data.contact_basis_n_a, wp.from_numpy(contact_basis_n_a_np, dtype=wp.spatial_vector))
                wp.copy(data.contact_dist, wp.from_numpy(contact_dist_np, dtype=wp.float32))
                wp.copy(data.contact_friction_coeff, wp.from_numpy(contact_friction_coeff_np, dtype=wp.float32))

            # Store for verification
            gt = {
                "body_q": body_q_np,
                "body_q_prev": body_q_prev_np,
                "body_u": body_u_np,
                "body_u_prev": body_u_prev_np,
                "body_f": body_f_np,
                "joint_target_pos": joint_target_pos_np,
                "joint_target_vel": joint_target_vel_np,
                "body_lambda": body_lambda_np,
                "body_lambda_prev": body_lambda_prev_np,
                "constraint_active_mask": constraint_active_mask_np,
                "constraint_body_idx": constraint_body_idx_np,
                "contact_body_a": contact_body_a_np,
                "contact_body_b": contact_body_b_np,
                "contact_point_a": contact_point_a_np,
                "contact_point_b": contact_point_b_np,
                "contact_basis_n_a": contact_basis_n_a_np,
                "contact_dist": contact_dist_np,
                "contact_friction_coeff": contact_friction_coeff_np,
            }
            ground_truth_history.append(gt)

            # ACT: Save to buffer
            buffer.save_step(t, data)

        # 2. Wipe EngineData (Set to zeros)
        data.body_q.zero_()
        data.body_q_prev.zero_()
        data.body_u.zero_()
        data.body_u_prev.zero_()
        data.body_f.zero_()
        data.joint_target_pos.zero_()
        data.joint_target_vel.zero_()
        data._body_lambda.zero_()
        data._body_lambda_prev.zero_()
        data._constraint_active_mask.zero_()
        data._constraint_body_idx.zero_()
        data.contact_body_a.zero_()
        data.contact_body_b.zero_()
        data.contact_point_a.zero_()
        data.contact_point_b.zero_()
        data.contact_basis_n_a.zero_()
        data.contact_dist.zero_()
        data.contact_friction_coeff.zero_()

        # 3. Verify Loading
        for t in range(num_steps):
            # ACT: Load from buffer
            buffer.load_step(t, data)

            # ASSERT: Compare against ground truth
            gt = ground_truth_history[t]

            for key, val in gt.items():
                # Map keys to EngineData attributes
                attr_name = key
                if key == "body_lambda": attr_name = "_body_lambda"
                if key == "body_lambda_prev": attr_name = "_body_lambda_prev"
                if key == "constraint_active_mask": attr_name = "_constraint_active_mask"
                if key == "constraint_body_idx": attr_name = "_constraint_body_idx"
                if key == "contact_friction_coeff": attr_name = "contact_friction_coeff"
                
                loaded_val = getattr(data, attr_name).numpy()
                
                if val.dtype == np.int32:
                    np.testing.assert_array_equal(loaded_val, val, err_msg=f"{key} mismatch at step {t}")
                else:
                    np.testing.assert_array_almost_equal(loaded_val, val, decimal=5, err_msg=f"{key} mismatch at step {t}")

    def test_step_isolation(self):
        """
        Ensures that saving to step 1 does not overwrite data at step 0.
        Crucial for strided memory bugs.
        """
        dims = self.create_dummy_dims()
        buffer = TrajectoryBuffer(dims, num_steps=2, device=self.device)
        data = self.create_dummy_data(dims)

        # Fill Step 0 with 1.0
        data._body_lambda.fill_(1.0)
        buffer.save_step(0, data)

        # Fill Step 1 with 2.0
        data._body_lambda.fill_(2.0)
        buffer.save_step(1, data)

        # Verify Step 0 is still 1.0
        data._body_lambda.zero_()
        buffer.load_step(0, data)

        avg_val = np.mean(data._body_lambda.numpy())
        self.assertAlmostEqual(
            avg_val, 1.0, places=5, msg="Step 0 lambda data was corrupted by Step 1 save!"
        )

        # Verify Step 1 is 2.0
        buffer.load_step(1, data)
        avg_val = np.mean(data._body_lambda.numpy())
        self.assertAlmostEqual(
            avg_val, 2.0, places=5, msg="Step 1 lambda data failed to save/load correctly."
        )


if __name__ == "__main__":
    unittest.main()
