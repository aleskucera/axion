from pathlib import Path
from typing import Union
import os.path as osp
import sys

import numpy as np
from tqdm import tqdm
import h5py
from h5py import Group

import torch
from torch_geometric.data import InMemoryDataset, HeteroData

from axion.gnn.graph_builder import (
    NODE_FEATURE_DIMS,
    EDGE_FEATURE_DIMS,
    OUTPUT_FEATURE_DIMS,
    build_graph,
)

VALID_THRESHOLD = 1e3


class AxionDatasetGNN(InMemoryDataset):
    """
    Args:
        root: path to directory containing HDF5 files for multiple passes
    """

    def __init__(self, root: Union[str, Path]):
        super().__init__(root)
        self.load(self.processed_paths[0], HeteroData)
        if osp.exists(self.processed_paths[1]):
            self.stats = torch.load(self.processed_paths[1], weights_only=False)
        else:
            self.stats = None

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt", "statistics.pt"]

    def process(self) -> None:
        graphs = []
        total_invalid = 0
        total_valid = 0
        for path in tqdm(
            sorted(Path(self.raw_dir).iterdir(), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        ):
            if not path.suffix == ".h5":
                continue
            with h5py.File(path, "r") as f:
                num_steps = f["data"]["body_vel"].shape[0]
                for step in np.arange(num_steps, step=max(10, num_steps // 10)):
                    graph_new = self.construct_graph(f, step)
                    graphs.extend(graph_new)
                    valid = len(self._valid_world_indices(f["data"], step, f["dims"]))
                    total_valid += valid
                    total_invalid += f["dims"]["num_worlds"][()] - valid
        print(f"Out of {total_valid + total_invalid} worlds {total_invalid} were invalid")

        stats = calculate_statistics(graphs)
        self.save(graphs, self.processed_paths[0])
        torch.save(stats, self.processed_paths[1])

    def construct_graph(self, file: Group, step: int) -> list[HeteroData]:
        model = file["model"]
        data = file["data"]
        dims = file["dims"]

        world_indices = self._valid_world_indices(data, step, dims)
        if not len(world_indices) > 0:
            return []

        num_bodies = dims["body_count"][()]
        device = torch.device("cpu")

        body_vel = torch.tensor(data["body_vel"][step][world_indices], dtype=torch.float32)
        body_vel_prev = torch.tensor(
            data["body_vel_prev"][step][world_indices], dtype=torch.float32
        )
        body_mass = torch.tensor(model["body_mass"][world_indices], dtype=torch.float32).unsqueeze(
            2
        )
        ext_force = torch.tensor(data["ext_force"][step][world_indices], dtype=torch.float32)
        body_pose_prev = torch.tensor(
            data["body_pose_prev"][step][world_indices], dtype=torch.float32
        )
        body_inertia = torch.tensor(model["body_inertia"][world_indices], dtype=torch.float32)
        body_com = torch.tensor(model["body_com"][world_indices], dtype=torch.float32)

        contact_count = torch.tensor(data["contact_count"][step][world_indices], dtype=torch.long)
        contact_shape0 = torch.tensor(data["contact_shape0"][step][world_indices], dtype=torch.long)
        contact_shape1 = torch.tensor(data["contact_shape1"][step][world_indices], dtype=torch.long)
        contact_point0 = torch.tensor(
            data["contact_point0"][step][world_indices], dtype=torch.float32
        )
        contact_point1 = torch.tensor(
            data["contact_point1"][step][world_indices], dtype=torch.float32
        )
        shape_material_mu = torch.tensor(
            model["shape_material_mu"][world_indices], dtype=torch.float32
        )
        contact_thickness0 = torch.tensor(
            data["contact_thickness0"][step][world_indices], dtype=torch.float32
        )
        contact_thickness1 = torch.tensor(
            data["contact_thickness1"][step][world_indices], dtype=torch.float32
        )
        contact_normal = torch.tensor(
            data["contact_normal"][step][world_indices], dtype=torch.float32
        )
        shape_body = torch.tensor(model["shape_body"][world_indices], dtype=torch.long)
        world_indices_tensor = torch.tensor(world_indices, dtype=torch.long)

        graph = build_graph(
            body_vel_prev,
            body_mass,
            ext_force,
            body_pose_prev,
            body_inertia,
            body_com,
            contact_count,
            contact_point0,
            contact_point1,
            contact_normal,
            contact_shape0,
            contact_shape1,
            shape_material_mu,
            contact_thickness0,
            contact_thickness1,
            num_bodies,
            device,
            shape_body,
            body_vel_next=body_vel,
            world_indices=world_indices_tensor,
        )

        return [graph]

    def _valid_world_indices(self, data: Group, step: int, dims: Group) -> np.ndarray:
        """Return a boolean array [W] that is True for worlds with finite, bounded data."""
        W = dims["num_worlds"][()]
        valid = np.ones(W, dtype=bool)
        for key in ("body_vel", "body_vel_prev", "body_pose"):
            arr = data[key][step][:]
            flat = arr.reshape(W, -1)
            valid &= ~np.any(np.isnan(flat) | (np.abs(flat) > VALID_THRESHOLD), axis=1)
        world_indices: np.ndarray = np.where(valid)[0]
        return world_indices


def calculate_statistics(graphs: list[HeteroData]) -> dict:
    stats = {"nodes": {}, "edges": {}}
    for node_type, dims in NODE_FEATURE_DIMS.items():
        if not dims > 0:
            continue
        all_x = [g[node_type].x for g in graphs if g[node_type].num_nodes > 0]
        if all_x:
            all_x = torch.cat(all_x, dim=0)
            mean = all_x.mean(dim=0)
            std = all_x.std(dim=0)
            std[std < 1e-6] = 1.0
            stats["nodes"][node_type] = {"mean": mean, "std": std}

    all_edge_types = set()
    for g in graphs:
        all_edge_types.update(g.edge_types)
    for edge_type in all_edge_types:
        if not EDGE_FEATURE_DIMS[edge_type] > 0:
            continue
        all_attr = [
            g[edge_type].edge_attr
            for g in graphs
            if edge_type in g.edge_attr_dict and g[edge_type].num_edges > 0
        ]
        if all_attr:
            all_attr = torch.cat(all_attr, dim=0)
            mean = all_attr.mean(dim=0)
            std = all_attr.std(dim=0)
            std[std < 1e-6] = 1.0
            stats["edges"]["_".join(edge_type)] = {"mean": mean, "std": std}
    return stats


if __name__ == "__main__":
    AxionDatasetGNN(root=sys.argv[1])
