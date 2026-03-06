from pathlib import Path
from typing import Union, Any
import os.path as osp
from collections import defaultdict
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import h5py
from h5py import Group

import torch
from torch_geometric.data import InMemoryDataset, HeteroData


NODE_FEATURE_DIMS = {"object": 22, "contact_point": 0, "floor": 0}
EDGE_FEATURE_DIMS = {
    ("object", "inter_object", "contact_point"): 4,
    ("floor", "inter_object", "contact_point"): 4,
    ("contact_point", "contact", "contact_point"): 5,
    ("object", "fixed_joint", "object"): 12,
    ("object", "revolute_joint", "object"): 8,
    ("object", "prismatic_joint", "object"): 8,
    ("floor", "fixed_joint", "object"): 12,
    ("floor", "revolute_joint", "object"): 8,
    ("floor", "prismatic_joint", "object"): 8,
}
OUTPUT_FEATURE_DIMS = {
    "object": 6,
    ("contact_point", "contact", "contact_point"): 3,
    ("object", "fixed_joint", "object"): 3,
    ("object", "revolute_joint", "object"): 2,
    ("object", "prismatic_joint", "object"): 2,
    ("floor", "fixed_joint", "object"): 3,
    ("floor", "revolute_joint", "object"): 2,
    ("floor", "prismatic_joint", "object"): 2,
}


def norm(x: Any):
    return torch.norm(torch.tensor(x, dtype=torch.float32))


class DatasetSim2D(InMemoryDataset):
    """
    Args:
        root: path to directory containing HDF5 files for multiple passes
    """

    def __init__(self, root: Union[str, Path]):
        super().__init__(root)
        self.load(self.processed_paths[0], HeteroData)
        if osp.exists(self.processed_paths[1]):
            self.stats = torch.load(self.processed_paths[1])
        else:
            self.stats = None

    @property
    def processed_file_names(self):
        return ["data.pt", "statistics.pt"]

    def process(self) -> None:
        graphs = []
        for path in tqdm(
            sorted(Path(self.raw_dir).iterdir(), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        ):
            if not path.suffix == ".h5":
                continue
            with h5py.File(path, "r") as f:
                for step in [0]:  # take first step, but more steps can be added
                    graphs = self.construct_graphs(f, step)
                    graphs.extend(graphs)
        stats = self.calculate_statistics(graphs)

        self.save(graphs, self.processed_paths[0])
        torch.save(stats, self.processed_paths[1])

    # ---- Dataset definition ----
    def construct_graphs(self, file: Group, step: int) -> list[HeteroData]:
        model = file["model"]
        data = file["data"]
        dims = file["dims"]
        graphs = []
        for world in range(dims["num_worlds"][()]):
            graph = HeteroData()
            num_nodes = 0
            # -- Objects --
            num_nodes += self.create_object_nodes(graph, step, world, model, data, dims)
            # -- Floor --
            num_nodes += self.create_floor_node(graph, step, world, model, data, dims)
            # -- Contacts --
            num_nodes += self.create_contacts(graph, step, world, model, data, dims)
            # --Joints --
            self.create_joint_edges(graph, step, world, model, data, dims)
            # -- Add graph --
            graph.num_nodes = num_nodes
            graphs.append(graph)
        return graphs

    def create_object_nodes(
        self, graph: HeteroData, step: int, world: int, model: Group, data: Group, dims: Group
    ) -> None:
        def _transform_toi(quat, toi):
            R = Rotation.from_quat(quat).as_matrix()
            return R @ toi @ R.T

        num_bodies = dims["body_count"][()]
        nodes_object = []
        preds_object = []
        for obj in range(num_bodies):
            nodes_object.append(
                [
                    *data["body_vel_prev"][step][world][obj],
                    model["body_mass"][world][obj],
                    *data["ext_force"][step][world][obj],
                    *_transform_toi(
                        data["body_pose_prev"][step][world][obj][3:],
                        model["body_inertia"][world][obj],
                    ).flatten(),
                ]
            )
            preds_object.append(
                [
                    *data["body_vel"][step][world][obj].tolist(),
                ]
            )
        graph["object"].x = torch.tensor(nodes_object, dtype=torch.float32)
        graph["object"].y = torch.tensor(preds_object, dtype=torch.float32)
        return num_bodies

    def create_floor_node(
        self, graph: HeteroData, step: int, world: int, model: Group, data: Group, dims: Group
    ) -> None:
        graph["floor"].x = torch.zeros((1, 0), dtype=torch.float32)
        return 1

    def create_contacts(
        self, graph: HeteroData, step: int, world: int, model: Group, data: Group, dims: Group
    ) -> None:
        def _find_global_index(body_idx, point_global):
            nonlocal points_cnt
            dists = np.linalg.norm(
                np.array(point_pos[point_body == body_idx]) - point_global, axis=1
            )
            if len(dists) > 0 and dists[np.argmin(dists)] < 1e-3:
                global_index = point_idx[point_body == body_idx][np.argmin(dists)]
                new_node = False
            else:
                global_index = points_cnt
                point_body[points_cnt] = body_idx
                point_pos[points_cnt] = point_global
                points_cnt += 1
                new_node = True
            return global_index, new_node

        def _transform(pos_quat: np.array, point: np.array):
            assert len(pos_quat) == 7
            return Rotation.from_quat(pos_quat[3:]).apply(point) + pos_quat[:3]

        nodes_contact = []

        attrs_object_contact = []
        indices_object_contact = []

        attrs_floor_contact = []
        indices_floor_contact = []

        attrs_contact_contact = []
        indices_contact_contact = []
        preds_contact_contact = []

        points_cnt = 0
        point_idx = np.arange(2 * data["contact_count"][step][world], dtype=int)
        point_body = np.zeros(2 * data["contact_count"][step][world], dtype=int)
        point_pos = np.inf * np.ones((2 * data["contact_count"][step][world], 3), dtype=float)
        for i in range(data["contact_count"][step][world]):
            contact_indices = [None, None]
            contact_attributes = [None, None]
            contact_preds = [None, None]
            for j in [0, 1]:
                shape_idx = data[f"contact_shape{j}"][step][world][i]
                body_idx = self._resolve_body_indices(shape_idx, model["shape_body"][world])
                if body_idx >= 0:
                    body_pose = data["body_pose"][step][world][body_idx]
                else:
                    body_pose = np.zeros(7, dtype=float)
                    body_pose[-1] = 1.0
                point = data[f"contact_point{j}"][step][world][i]
                point_global = _transform(body_pose, point)
                contact_idx, new_node = _find_global_index(body_idx, point_global)

                normal = (-1 if j else 1) * data["contact_normal"][step][world][i][()]
                mu = model["shape_material_mu"][world][shape_idx]
                com = model["body_com"][world][body_idx]
                thicc = data[f"contact_thickness{j}"][step][world][i]
                point_adj = point_global - (thicc * normal)
                lever = point_adj - _transform(body_pose, com)
                lever_norm = np.linalg.norm(lever)

                if new_node:
                    nodes_contact.append([])
                if body_idx == -1:
                    indices_floor_contact.append([0, contact_idx])
                    indices_floor_contact.append([contact_idx, 0])
                    attrs_floor_contact.append([*lever, lever_norm])
                    attrs_floor_contact.append([*-lever, lever_norm])
                else:
                    indices_object_contact.append([body_idx, contact_idx])
                    indices_object_contact.append([contact_idx, body_idx])
                    attrs_object_contact.append([*lever, lever_norm])
                    attrs_object_contact.append([*-lever, lever_norm])
                contact_indices[j] = contact_idx
                contact_attributes[j] = [normal, point_adj, mu]
                contact_preds[j] = []

            mu = (contact_attributes[0][2] + contact_attributes[1][2]) / 2
            dist = np.dot(-normal, contact_attributes[0][1] - contact_attributes[1][1])
            assert dist >= -0.0001, f"Distance is not greater than 0.0, current value is: {dist}"

            indices_contact_contact.append(contact_indices)
            indices_contact_contact.append(contact_indices[::-1])
            attrs_contact_contact.append([*contact_attributes[0][0], dist, mu])
            attrs_contact_contact.append([*contact_attributes[1][0], dist, mu])
            preds_contact_contact.append([])
            preds_contact_contact.append([])

        graph["contact_point"].x = torch.tensor(nodes_contact, dtype=torch.float32)
        self._assign_edge_data(
            graph,
            ("object", "inter_object", "contact_point"),
            indices_object_contact,
            attrs_object_contact,
            EDGE_FEATURE_DIMS[("object", "inter_object", "contact_point")],
        )
        self._assign_edge_data(
            graph,
            ("floor", "inter_object", "contact_point"),
            indices_floor_contact,
            attrs_floor_contact,
            EDGE_FEATURE_DIMS[("floor", "inter_object", "contact_point")],
        )
        self._assign_edge_data(
            graph,
            ("contact_point", "contact", "contact_point"),
            indices_contact_contact,
            attrs_contact_contact,
            EDGE_FEATURE_DIMS[("contact_point", "contact", "contact_point")],
            preds_contact_contact,
            OUTPUT_FEATURE_DIMS[("contact_point", "contact", "contact_point")],
        )
        return points_cnt

    def create_joint_edges(
        self, graph: HeteroData, step: int, world: int, model: Group, data: Group, dims: Group
    ) -> None:
        pass

    # ---- Helper functions ----
    def _resolve_body_indices(self, shape0: int, shape_body: np.ndarray):
        body0 = -1
        if shape0 >= 0:
            body0 = shape_body[shape0]
        return body0

    def _assign_edge_data(
        self, graph, edge_type, indices, attrs, attr_dim, preds=None, pred_dim=None
    ):
        if len(indices) > 0:
            graph[edge_type].edge_index = torch.tensor(indices, dtype=torch.long).T
            graph[edge_type].edge_attr = torch.tensor(attrs, dtype=torch.float32)
            if not preds is None:
                graph[edge_type].y = torch.tensor(preds, dtype=torch.float32)
        else:
            graph[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph[edge_type].edge_attr = torch.zeros((0, attr_dim), dtype=torch.float32)
            if not pred_dim is None:
                graph[edge_type].y = torch.zeros((0, pred_dim), dtype=torch.float32)

    # ---- Statistics calculation ----
    def calculate_statistics(self, graphs):
        stats = {"nodes": {}, "edges": {}}
        for node_type in NODE_FEATURE_DIMS.keys():
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
    DatasetSim2D(root="data/gnn_data/test_dataset")
