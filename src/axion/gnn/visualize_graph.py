"""Visualise a world-batched HeteroData graph produced by AxionDatasetGNN.

Each simulation world is laid out independently and then spatially offset so
the disconnected sub-graphs are easy to distinguish.

Usage
-----
    python visualize_batched_graph.py --dataset_root <path/to/dataset> --idx 0
"""
import argparse
import math

import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from torch_geometric.data import HeteroData

from axion.gnn.dataset import AxionDatasetGNN


# ──────────────────────────────────────────────────────────────────────────────
# Color palette (mirrors visualize_graph_2d.py)
# ──────────────────────────────────────────────────────────────────────────────

NODE_COLORS = {
    "object": "#006EFF",
    "floor": "#32532E",
    "contact_point": "#C0C0C0",
}

EDGE_COLORS = {
    ("object", "inter_object", "contact_point"): ("#FF4444", "solid"),
    ("contact_point", "inter_object", "object"): ("#FF8800", "solid"),
    ("floor", "inter_object", "contact_point"): ("#FC802E", "solid"),
    ("contact_point", "contact", "contact_point"): ("#AA00FF", "solid"),
    ("object", "fixed_joint", "object"): ("#8A2BE2", "dashed"),
    ("object", "revolute_joint", "object"): ("#FFD700", "dashed"),
    ("object", "prismatic_joint", "object"): ("#20B2AA", "dashed"),
    ("floor", "fixed_joint", "object"): ("#8A2BE2", "dashed"),
    ("floor", "revolute_joint", "object"): ("#FFD700", "dashed"),
    ("floor", "prismatic_joint", "object"): ("#20B2AA", "dashed"),
}

DEFAULT_NODE_COLOR = "#DDDDDD"
DEFAULT_EDGE_COLOR = ("#000000", "solid")

WORLD_PALETTE = [
    "#E6F0FF", "#FFF5E6", "#E6FFE6", "#FFE6FF",
    "#FFFFE6", "#E6FFFF", "#FFE6E6", "#F0E6FF",
]


# ──────────────────────────────────────────────────────────────────────────────
# Build NetworkX graph
# ──────────────────────────────────────────────────────────────────────────────

def _build_nx_graph(data: HeteroData):
    """Convert HeteroData to a NetworkX graph with global node IDs.

    Each node carries ``type``, ``local_idx``, ``world``, and ``label``
    attributes.  Edges carry ``edge_type``.
    """
    G = nx.MultiDiGraph()
    node_mapping: dict[tuple, int] = {}  # (node_type, local_idx) → global_id
    gid = 0

    for node_type in data.node_types:
        num = data[node_type].num_nodes
        world_attr = getattr(data[node_type], "world", None)
        for local in range(num):
            world = int(world_attr[local]) if world_attr is not None else 0
            label = f"{node_type[0].upper()}{local}"
            G.add_node(gid, type=node_type, local=local, world=world, label=label)
            node_mapping[(node_type, local)] = gid
            gid += 1

    for edge_type_tuple in data.edge_types:
        src_type, _, dst_type = edge_type_tuple
        ei = data[edge_type_tuple].edge_index
        if ei.numel() == 0:
            continue
        for k in range(ei.shape[1]):
            src_g = node_mapping.get((src_type, int(ei[0, k])))
            dst_g = node_mapping.get((dst_type, int(ei[1, k])))
            if src_g is not None and dst_g is not None:
                G.add_edge(src_g, dst_g, edge_type=edge_type_tuple)

    return G, node_mapping


# ──────────────────────────────────────────────────────────────────────────────
# Per-world layout with spatial offset
# ──────────────────────────────────────────────────────────────────────────────

def _compute_layout(G: nx.MultiDiGraph, world_order: list[int], spacing: float = 6.0) -> dict:
    """Compute spring layout per world, then offset each world's positions.

    Worlds are arranged in a grid: sqrt(N) columns, ceil(N/cols) rows.
    Grid positions are assigned by sequential index so filtered worlds leave no gaps.
    """
    num_worlds = len(world_order)
    cols = max(1, math.isqrt(num_worlds))

    # Group nodes by original world id
    world_nodes: dict[int, list[int]] = {}
    for n, attr in G.nodes(data=True):
        w = attr["world"]
        world_nodes.setdefault(w, []).append(n)

    pos: dict[int, tuple[float, float]] = {}

    for slot, w in enumerate(world_order):
        nodes = world_nodes.get(w, [])
        sub = G.subgraph(nodes)
        if len(nodes) == 0:
            continue
        if len(nodes) == 1:
            sub_pos = {nodes[0]: (0.0, 0.0)}
        else:
            sub_pos = nx.spring_layout(sub, k=2.5, iterations=60, seed=w + 42)

        # Grid offset based on sequential slot, not original world id
        col = slot % cols
        row = slot // cols
        ox = col * spacing
        oy = -row * spacing

        for n, (x, y) in sub_pos.items():
            pos[n] = (x + ox, y + oy)

    return pos


# ──────────────────────────────────────────────────────────────────────────────
# Main visualisation function
# ──────────────────────────────────────────────────────────────────────────────

def visualize_batched_graph(data: HeteroData, title: str = "Batched Graph") -> None:
    """Draw a world-batched HeteroData graph with spatially separated worlds."""
    if not isinstance(data, HeteroData):
        raise TypeError("Expected HeteroData")

    world_attr = getattr(data.get("object", data[data.node_types[0]]), "world", None)
    unique_worlds = sorted(world_attr.unique().tolist()) if world_attr is not None and len(world_attr) > 0 else [0]
    num_worlds = len(unique_worlds)

    G, _ = _build_nx_graph(data)
    pos = _compute_layout(G, unique_worlds)
    labels = {n: attr["label"] for n, attr in G.nodes(data=True)}

    fig, ax = plt.subplots(figsize=(max(10, num_worlds * 5), 8))

    NODE_SIZE = 800
    ARROW_SIZE = 20

    # ── World background patches ──
    world_node_positions: dict[int, list] = {}
    for n, attr in G.nodes(data=True):
        world_node_positions.setdefault(attr["world"], []).append(pos[n])

    for w, pts in world_node_positions.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 0.8
        bg_color = WORLD_PALETTE[w % len(WORLD_PALETTE)]
        rect = plt.Rectangle(
            (min(xs) - pad, min(ys) - pad),
            max(xs) - min(xs) + 2 * pad,
            max(ys) - min(ys) + 2 * pad,
            linewidth=1.5,
            edgecolor="gray",
            facecolor=bg_color,
            alpha=0.35,
            zorder=0,
        )
        ax.add_patch(rect)
        cx = (min(xs) + max(xs)) / 2
        cy = max(ys) + pad * 0.6
        ax.text(cx, cy, f"World {w}", ha="center", va="bottom", fontsize=9, color="gray")

    # ── Nodes ──
    seen_node_types: set[str] = set()
    for node_type in data.node_types:
        nodelist = [n for n, attr in G.nodes(data=True) if attr["type"] == node_type]
        if not nodelist:
            continue
        seen_node_types.add(node_type)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_color=NODE_COLORS.get(node_type, DEFAULT_NODE_COLOR),
            node_size=NODE_SIZE,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.5,
            ax=ax,
        )

    # ── Edges (batched by style) ──
    edge_batches: dict[str, dict] = {}
    for u, v, attr in G.edges(data=True):
        et = attr["edge_type"]
        color, style = EDGE_COLORS.get(et, DEFAULT_EDGE_COLOR)
        key = str(et)
        if key not in edge_batches:
            edge_batches[key] = {"edge_type": et, "color": color, "style": style, "edges": []}
        edge_batches[key]["edges"].append((u, v))

    for batch in edge_batches.values():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=batch["edges"],
            edge_color=batch["color"],
            style=batch["style"],
            width=1.8,
            arrowsize=ARROW_SIZE,
            node_size=NODE_SIZE,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", ax=ax)

    # ── Legend ──
    legend_handles = []
    for nt in sorted(seen_node_types):
        legend_handles.append(
            mpatches.Patch(color=NODE_COLORS.get(nt, DEFAULT_NODE_COLOR), label=f"Node: {nt}")
        )
    for batch in edge_batches.values():
        et = batch["edge_type"]
        label = " → ".join(et) if isinstance(et, tuple) else str(et)
        legend_handles.append(
            Line2D(
                [0], [0],
                color=batch["color"],
                linestyle=batch["style"],
                linewidth=1.8,
                label=f"Edge: {label}",
            )
        )

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise a world-batched graph from AxionDatasetGNN"
    )
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of the processed dataset")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index of the graph within the dataset")
    args = parser.parse_args()

    dataset = AxionDatasetGNN(root=args.dataset_root)
    graph = dataset[args.idx]
    visualize_batched_graph(graph, title=f"Dataset: {args.dataset_root}  |  idx={args.idx}")
