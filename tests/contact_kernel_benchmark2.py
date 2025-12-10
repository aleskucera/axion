import csv
import statistics
import time

import numpy as np
import nvtx  # pip install nvtx
import warp as wp
from axion.constraints import contact_constraint_kernel
from axion.types import ContactInteraction
from axion.types import SpatialInertia

wp.init()


class BenchmarkRunner:
    def __init__(self, device):
        self.device = device
        self.csv_data = []

        self.start_event = wp.Event(enable_timing=True)
        self.end_event = wp.Event(enable_timing=True)

    def generate_topology(self, n_w, n_b, n_c, scenario="random"):
        """
        Generates indices based on contention scenarios.
        scenario="random": Standard uniform random.
        scenario="high_contention": All contacts interact with Body 0.
        scenario="one_to_one": Contact[i] interacts with Body[i] and Body[i+1].
        """
        if scenario == "high_contention":
            # Everyone touches body 0
            body_a = np.zeros((n_w, n_c), dtype=np.int32)
            body_b = np.random.randint(1, n_b, size=(n_w, n_c), dtype=np.int32)
        elif scenario == "one_to_one":
            # Minimal atomic collision (best case)
            idx = np.arange(n_c) % n_b
            body_a = np.tile(idx, (n_w, 1)).astype(np.int32)
            body_b = np.tile((idx + 1) % n_b, (n_w, 1)).astype(np.int32)
        else:
            # Random
            body_a = np.random.randint(0, n_b, size=(n_w, n_c), dtype=np.int32)
            body_b = np.random.randint(0, n_b, size=(n_w, n_c), dtype=np.int32)

        return body_a, body_b

    def setup_data(self, n_w, n_b, n_c, scenario="random"):
        """Allocates Warp arrays."""
        # --- 1. Topology Generation ---
        body_a_np, body_b_np = self.generate_topology(n_w, n_b, n_c, scenario)

        # --- 2. Dummy Physics Data ---
        # We construct interactions manually or using a simplified struct mapper
        # For benchmarking, we just need the memory layout to be correct.
        # Warp allows initializing structs from dicts or flat arrays if careful,
        # but here we replicate your list logic optimized.

        interactions_list = []
        for w in range(n_w):
            for c in range(n_c):
                inter = ContactInteraction()
                inter.is_active = True if (c % 10 != 0) else False  # 90% active
                inter.body_a_idx = int(body_a_np[w, c])
                inter.body_b_idx = int(body_b_np[w, c])
                inter.penetration_depth = 0.05
                inter.friction_coeff = 0.5
                inter.basis_a.normal = wp.spatial_vector(0, 1, 0, 0, 0, 0)  # Simplified
                inter.basis_b.normal = wp.spatial_vector(0, 1, 0, 0, 0, 0)
                interactions_list.append(inter)

        # --- 3. Warp Allocation ---
        data = {
            "body_u": wp.zeros((n_w, n_b), dtype=wp.spatial_vector, device=self.device),
            "body_u_prev": wp.zeros((n_w, n_b), dtype=wp.spatial_vector, device=self.device),
            "body_lambda_n": wp.zeros((n_w, n_c), dtype=wp.float32, device=self.device),
            "interactions": wp.array(
                interactions_list, dtype=ContactInteraction, device=self.device
            ).reshape((n_w, n_c)),
            "body_M_inv": wp.zeros((n_w, n_b), dtype=SpatialInertia, device=self.device),
            # Constants
            "dt": 0.016,
            "stabilization_factor": 0.1,
            "fb_alpha": 1.0,
            "fb_beta": 1.0,
            "compliance": 0.0,
            # Outputs
            "h_d": wp.zeros((n_w, n_b), dtype=wp.spatial_vector, device=self.device),
            "h_n": wp.zeros((n_w, n_c), dtype=wp.float32, device=self.device),
            "J_hat_n_values": wp.zeros((n_w, n_c, 2), dtype=wp.spatial_vector, device=self.device),
            "C_n_values": wp.zeros((n_w, n_c), dtype=wp.float32, device=self.device),
            "s_n": wp.zeros((n_w, n_c), dtype=wp.float32, device=self.device),
        }
        return data

    def run_kernel(self, kernel, data, dim):
        wp.launch(
            kernel=kernel,
            dim=dim,
            inputs=[
                data["body_u"],
                data["body_u_prev"],
                data["body_lambda_n"],
                data["interactions"],
                data["body_M_inv"],
                data["dt"],
                data["stabilization_factor"],
                data["fb_alpha"],
                data["fb_beta"],
                data["compliance"],
                data["h_d"],
                data["h_n"],
                data["J_hat_n_values"],
                data["C_n_values"],
                data["s_n"],
            ],
            device=self.device,
        )

    def benchmark(self, kernel, nw, nb, nc, use_graph=True, scenario="random"):
        data = self.setup_data(nw, nb, nc, scenario)
        dim = (nw, nc)

        # --- Warmup ---
        self.run_kernel(kernel, data, dim)
        wp.synchronize()

        # --- Capture Graph (Optional) ---
        graph = None
        if use_graph and self.device.is_cuda:
            wp.capture_begin(device=self.device)
            self.run_kernel(kernel, data, dim)
            graph = wp.capture_end(device=self.device)

        # --- Timing Loop ---
        repeats = 10
        iters = 50
        timings = []

        for _ in range(repeats):
            # NVTX Range helps see this specific block in Nsight Systems
            with nvtx.annotate(f"Bench {nw}x{nc} {scenario}", color="green"):
                wp.record_event(self.start_event)

                if graph:
                    for _ in range(iters):
                        wp.capture_launch(graph)
                else:
                    for _ in range(iters):
                        self.run_kernel(kernel, data, dim)

                wp.record_event(self.end_event)
                wp.synchronize()  # Wait for GPU

                # Elapsed time in ms
                elapsed_ms = wp.get_event_elapsed_time(self.start_event, self.end_event)
                timings.append(elapsed_ms / iters)

        avg_ms = statistics.mean(timings)
        std_ms = statistics.stdev(timings)

        # --- Store Results ---
        result = {
            "N_WORLDS": nw,
            "N_BODIES": nb,
            "N_CONTACTS": nc,
            "Scenario": scenario,
            "Graph": use_graph,
            "Time_ms": avg_ms,
            "Std_ms": std_ms,
            "Contacts_Per_Sec": (nw * nc) / (avg_ms / 1000.0),
        }
        self.csv_data.append(result)

        print(
            f"[{scenario.ljust(15)}] {nw=}, {nc=}, Graph={use_graph} -> {avg_ms:.4f} ms | {result['Contacts_Per_Sec']/1e6:.2f} M_Contact/s"
        )

    def save(self, filename="results.csv"):
        if not self.csv_data:
            return
        keys = self.csv_data[0].keys()
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.csv_data)


if __name__ == "__main__":
    runner = BenchmarkRunner(wp.get_device())

    # Compare Atomic Contention
    print("--- Contention Sweep ---")
    runner.benchmark(contact_constraint_kernel, nw=1, nb=128, nc=100000, scenario="one_to_one")
    runner.benchmark(contact_constraint_kernel, nw=1, nb=128, nc=100000, scenario="random")
    runner.benchmark(contact_constraint_kernel, nw=1, nb=128, nc=100000, scenario="high_contention")

    # Compare Graph overhead (Critical for small counts)
    print("\n--- Graph Overhead Sweep ---")
    runner.benchmark(contact_constraint_kernel, nw=1, nb=64, nc=1000, use_graph=False)
    runner.benchmark(contact_constraint_kernel, nw=1, nb=64, nc=1000, use_graph=True)

    runner.save()
