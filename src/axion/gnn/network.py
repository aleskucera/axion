from typing import Dict, Tuple, Union, Any

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .dataset import (
    NODE_FEATURE_DIMS,
    EDGE_FEATURE_DIMS,
    OUTPUT_FEATURE_DIMS,
)


class MLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 128,
        hidden_layers: int = 2,
        output_norm: bool = True,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dims, hidden_dims))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims, output_dims))
        if output_norm:
            layers.append(nn.LayerNorm(output_dims))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(
        self, hidden_dims: int, hidden_layers: int, normalize: bool, stats: Dict[str, Any] = None
    ):
        super().__init__()
        self.mlp_nodes = nn.ModuleDict(
            {
                node_type: MLP(node_dim, hidden_dims, hidden_dims, hidden_layers, normalize)
                for node_type, node_dim in NODE_FEATURE_DIMS.items()
            }
        )
        self.mlp_edges = nn.ModuleDict(
            {
                "_".join(edge_type): MLP(
                    edge_dim, hidden_dims, hidden_dims, hidden_layers, normalize
                )
                for edge_type, edge_dim in EDGE_FEATURE_DIMS.items()
            }
        )

        if stats is not None:
            for node_type, s in stats["nodes"].items():
                self.register_buffer(f"mean_node_{node_type}", s["mean"])
                self.register_buffer(f"std_node_{node_type}", s["std"])

            for edge_key, s in stats["edges"].items():
                self.register_buffer(f"mean_edge_{edge_key}", s["mean"])
                self.register_buffer(f"std_edge_{edge_key}", s["std"])

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_attr_dict: Dict[Tuple, torch.Tensor]):
        x_dict_encoded = {}
        for node_type, x in x_dict.items():
            if hasattr(self, f"mean_node_{node_type}"):
                mean = getattr(self, f"mean_node_{node_type}")
                std = getattr(self, f"std_node_{node_type}")
                x = (x - mean) / std
            x_dict_encoded[node_type] = self.mlp_nodes[node_type](x)

        edge_attr_dict_encoded = {}
        for edge_type, edge_attr in edge_attr_dict.items():
            edge_key = "_".join(edge_type)
            if hasattr(self, f"mean_edge_{edge_key}"):
                mean = getattr(self, f"mean_edge_{edge_key}")
                std = getattr(self, f"std_edge_{edge_key}")
                edge_attr = (edge_attr - mean) / std
            edge_attr_dict_encoded[edge_type] = self.mlp_edges[edge_key](edge_attr)

        return x_dict_encoded, edge_attr_dict_encoded


class Decoder(nn.Module):
    def __init__(self, hidden_dims: int, hidden_layers: int):
        super().__init__()
        self.mlp_nodes = nn.ModuleDict()
        for key, out_dim in OUTPUT_FEATURE_DIMS.items():
            if type(key) != str:
                continue
            self.mlp_nodes[key] = MLP(
                hidden_dims,
                out_dim,
                hidden_dims,
                hidden_layers,
                False,
            )

        self.mlp_edges = nn.ModuleDict()
        for key, out_dim in OUTPUT_FEATURE_DIMS.items():
            if type(key) != tuple:
                continue
            edge_key = "_".join(key)
            self.mlp_edges[edge_key] = MLP(
                hidden_dims,
                out_dim,
                hidden_dims,
                hidden_layers,
                False,
            )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_attr_dict: Dict[Tuple, torch.Tensor]):
        nodes_decoded = {}
        for key, node_attr in x_dict.items():
            if key in self.mlp_nodes:
                nodes_decoded[key] = self.mlp_nodes[key](node_attr)
        edges_decoded = {}
        for edge_type, edge_attr in edge_attr_dict.items():
            edge_key = "_".join(edge_type)
            if edge_key in self.mlp_edges:
                edges_decoded[edge_type] = self.mlp_edges[edge_key](edge_attr)
        return nodes_decoded, edges_decoded


class InteractionNetwork(MessagePassing):
    def __init__(self, hidden_dims: int, hidden_layers: int, normalize: bool, aggr: str = "add"):
        super().__init__(aggr)
        self.mlp_node = MLP(2 * hidden_dims, hidden_dims, hidden_dims, hidden_layers, normalize)
        self.mlp_edge = MLP(3 * hidden_dims, hidden_dims, hidden_dims, hidden_layers, normalize)

    def edge_update(self, x_i, x_j, edge_attr):
        return self.mlp_edge(torch.cat([x_i, x_j, edge_attr], dim=-1)) + edge_attr

    def message(self, edge_attr):
        return edge_attr

    def update(self, aggr_out, x, has_incoming=None):
        if isinstance(x, tuple):
            x = x[1]
        x_updated = self.mlp_node(torch.cat([x, aggr_out], dim=-1)) + x
        if has_incoming is not None:
            x_updated = torch.where(has_incoming.unsqueeze(-1), x_updated, x)
        return x_updated

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        if isinstance(x, tuple):
            size = (x[0].size(0), x[1].size(0))
        else:
            size = (x.size(0), x.size(0))

        has_incoming = torch.zeros(size[1], dtype=torch.bool, device=edge_index.device)
        if edge_index.numel() > 0:
            has_incoming[edge_index[1]] = True

        edge_attr_updated = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        x_updated = self.propagate(
            edge_index, x=x, edge_attr=edge_attr_updated, size=size, has_incoming=has_incoming
        )
        return x_updated, edge_attr_updated


class Processor(nn.Module):
    def __init__(
        self,
        message_passes: int,
        hidden_dims: int,
        hidden_layers: int,
        normalize: bool,
        repetitions: int = 1,
    ):
        super().__init__()
        self.repetitions = repetitions
        self.processor_layers = nn.ModuleList()
        for _ in range(message_passes):
            layer_dict = nn.ModuleDict()
            for edge_type in EDGE_FEATURE_DIMS.keys():
                layer_dict["_".join(edge_type)] = InteractionNetwork(
                    hidden_dims, hidden_layers, normalize
                )
            self.processor_layers.append(layer_dict)

    def _run_layers(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str], torch.Tensor],
    ):
        for layer in self.processor_layers:
            x_res_aggr = {key: torch.zeros_like(x) for key, x in x_dict.items()}
            for edge_type in edge_index_dict.keys():
                src_type, _, dst_type = edge_type
                edge_index = edge_index_dict[edge_type]
                if edge_index.shape[1] == 0:
                    continue
                edge_attr = edge_attr_dict[edge_type]
                if src_type == dst_type:
                    x = x_dict[src_type]
                else:
                    x = (x_dict[src_type], x_dict[dst_type])
                x_updated, edge_attr_updated = layer["_".join(edge_type)](x, edge_index, edge_attr)
                edge_attr_dict[edge_type] = edge_attr_updated
                x_res_aggr[dst_type] += x_updated - x_dict[dst_type]
            for node_type, res_aggr in x_res_aggr.items():
                x_dict[node_type] = x_dict[node_type] + res_aggr
        return x_dict, edge_attr_dict

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str], torch.Tensor],
    ):
        for _ in range(self.repetitions):
            x_dict, edge_attr_dict = self._run_layers(x_dict, edge_index_dict, edge_attr_dict)
        return x_dict, edge_attr_dict


class AxionGNN(nn.Module):
    def __init__(
        self,
        message_passes: int,
        hidden_dims: int,
        hidden_layers: int,
        normalize: bool,
        stats: Dict[str, Any] = None,
        repetitions: int = 1,
    ):
        super().__init__()
        self.encoder = Encoder(hidden_dims, hidden_layers, normalize, stats=stats)
        self.processor = Processor(
            message_passes, hidden_dims, hidden_layers, normalize, repetitions
        )
        self.decoder = Decoder(hidden_dims, hidden_layers)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict, edge_attr_dict = self.encoder(x_dict, edge_attr_dict)
        x_dict, edge_attr_dict = self.processor(x_dict, edge_index_dict, edge_attr_dict)
        nodes_dict, edges_dict = self.decoder(x_dict, edge_attr_dict)
        return nodes_dict, edges_dict
