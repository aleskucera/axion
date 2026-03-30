"""Visualise a world-batched HeteroData graph produced by AxionDatasetGNN.

Each simulation world is laid out independently and then spatially offset so
the disconnected sub-graphs are easy to distinguish.

Usage
-----
    python visualize_batched_graph.py --dataset_root <path/to/dataset> --idx 0
    python visualize_batched_graph.py --dataset_root <path/to/dataset> --idx 0 --show_attrs
"""

import argparse
import math

import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, TextBox
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
    "#E6F0FF",
    "#FFF5E6",
    "#E6FFE6",
    "#FFE6FF",
    "#FFFFE6",
    "#E6FFFF",
    "#FFE6E6",
    "#F0E6FF",
]


# ──────────────────────────────────────────────────────────────────────────────
# Drag manager for independent annotation dragging
# ──────────────────────────────────────────────────────────────────────────────


class _AnnotationDragManager:
    """Allows each annotation to be dragged independently.

    Replaces matplotlib's built-in ``draggable()`` which uses pick events with
    overlapping hit areas and can fire on multiple annotations at once.
    """

    def __init__(self, fig: plt.Figure, annotations: list) -> None:
        self._annotations = annotations
        self._dragging = None
        self._press_data: tuple[float, float] | None = None  # data coords at press
        self._ann_origin: tuple[float, float] | None = None  # annotation xyann at press
        self._fig = fig
        self._cids = [
            fig.canvas.mpl_connect("button_press_event", self._on_press),
            fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
            fig.canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def disconnect(self) -> None:
        for cid in self._cids:
            self._fig.canvas.mpl_disconnect(cid)
        self._cids.clear()

    def _on_press(self, event) -> None:
        if event.inaxes is None or event.button != 1:
            return
        # Find the topmost annotation whose rendered bbox contains the click
        for ann in reversed(self._annotations):
            try:
                bbox = ann.get_window_extent()
            except Exception:
                continue
            if bbox.contains(event.x, event.y):
                self._dragging = ann
                self._press_data = (event.xdata, event.ydata)
                self._ann_origin = ann.xyann
                break

    def _on_motion(self, event) -> None:
        if self._dragging is None or event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._press_data[0]
        dy = event.ydata - self._press_data[1]
        self._dragging.xyann = (
            self._ann_origin[0] + dx,
            self._ann_origin[1] + dy,
        )
        self._dragging.figure.canvas.draw_idle()

    def _on_release(self, event) -> None:
        self._dragging = None


# ──────────────────────────────────────────────────────────────────────────────
# Build NetworkX graph
# ──────────────────────────────────────────────────────────────────────────────


def _build_nx_graph(data: HeteroData, show_attrs: bool = False):
    """Convert HeteroData to a NetworkX graph with global node IDs.

    Each node carries ``type``, ``local_idx``, ``world``, ``label``, and
    optionally ``feats`` (list of floats from ``x``).
    Edges carry ``edge_type`` and optionally ``feats`` (list of floats from
    ``edge_attr``).
    """
    G = nx.MultiDiGraph()
    node_mapping: dict[tuple, int] = {}  # (node_type, local_idx) → global_id
    gid = 0

    for node_type in data.node_types:
        num = data[node_type].num_nodes
        world_attr = getattr(data[node_type], "world", None)
        x_tensor = data[node_type].x if show_attrs else None
        y_tensor = getattr(data[node_type], "y", None) if show_attrs else None
        for local in range(num):
            world = int(world_attr[local]) if world_attr is not None else 0
            label = f"{node_type[0].upper()}{local}"
            feats: list[float] = []
            if x_tensor is not None and x_tensor.shape[1] > 0:
                feats = [round(float(v), 4) for v in x_tensor[local]]
            preds: list[float] = []
            if y_tensor is not None and y_tensor.shape[1] > 0:
                preds = [round(float(v), 4) for v in y_tensor[local]]
            G.add_node(
                gid, type=node_type, local=local, world=world, label=label, feats=feats, preds=preds
            )
            node_mapping[(node_type, local)] = gid
            gid += 1

    for edge_type_tuple in data.edge_types:
        src_type, _, dst_type = edge_type_tuple
        ei = data[edge_type_tuple].edge_index
        if ei.numel() == 0:
            continue
        ea = data[edge_type_tuple].edge_attr if show_attrs else None
        for k in range(ei.shape[1]):
            src_g = node_mapping.get((src_type, int(ei[0, k])))
            dst_g = node_mapping.get((dst_type, int(ei[1, k])))
            if src_g is not None and dst_g is not None:
                feats: list[float] = []
                if ea is not None and ea.shape[0] > k:
                    feats = [round(float(v), 4) for v in ea[k]]
                G.add_edge(src_g, dst_g, edge_type=edge_type_tuple, feats=feats)

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


def _draw_into_axes(
    fig: plt.Figure,
    ax: plt.Axes,
    data: HeteroData,
    title: str,
    show_attrs: bool,
) -> "_AnnotationDragManager | None":
    """Clear *ax* and draw *data* into it.  Returns the drag manager (or None)."""
    ax.cla()
    ax.axis("off")

    world_attr = getattr(data.get("object", data[data.node_types[0]]), "world", None)
    unique_worlds = (
        sorted(world_attr.unique().tolist())
        if world_attr is not None and len(world_attr) > 0
        else [0]
    )

    G, _ = _build_nx_graph(data, show_attrs=show_attrs)
    pos = _compute_layout(G, unique_worlds)
    labels = {n: attr["label"] for n, attr in G.nodes(data=True)}

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

    # ── Attribute annotations ──
    draggable_annotations = []
    if show_attrs:
        # Node features
        for n, attr in G.nodes(data=True):
            feats = attr.get("feats", [])
            preds = attr.get("preds", [])
            if not feats and not preds:
                continue
            parts = []
            if feats:
                parts.append("\n".join(str(v) for v in feats))
            if preds:
                sep = "─" * max(len(str(v)) for v in (feats or preds))
                parts.append(sep)
                parts.append("\n".join(str(v) for v in preds))
            text = "\n".join(parts)
            xn, yn = pos[n]
            ann = ax.annotate(
                text,
                xy=(xn, yn),
                xytext=(xn + 0.1, yn),
                textcoords="data",
                fontsize=7,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, linewidth=0.5),
                zorder=5,
            )
            draggable_annotations.append(ann)

        # Edge features — annotate at geometric midpoint, offset by edge key to
        # avoid overlapping labels on parallel edges
        for u, v, key, attr in G.edges(data=True, keys=True):
            feats = attr.get("feats", [])
            if not feats:
                continue
            pu, pv = pos[u], pos[v]
            dx, dy = pv[0] - pu[0], pv[1] - pu[1]
            length = max((dx**2 + dy**2) ** 0.5, 1e-6)
            perp_x, perp_y = -dy / length, dx / length
            offset = 0.15 * (key - 0.5)
            mx = (pu[0] + pv[0]) / 2 + perp_x * offset
            my = (pu[1] + pv[1]) / 2 + perp_y * offset
            text = "\n".join(str(v) for v in feats)
            ann = ax.annotate(
                text,
                xy=(mx, my),
                xytext=(mx, my),
                textcoords="data",
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8, linewidth=0.5
                ),
                zorder=5,
            )
            draggable_annotations.append(ann)

        _AnnotationDragManager(fig, draggable_annotations)

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
                [0],
                [0],
                color=batch["color"],
                linestyle=batch["style"],
                linewidth=1.8,
                label=f"Edge: {label}",
            )
        )

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    if draggable_annotations:
        return _AnnotationDragManager(fig, draggable_annotations)
    return None


def visualize_batched_graph(
    data: HeteroData,
    title: str = "Batched Graph",
    show_attrs: bool = False,
) -> None:
    """Draw a world-batched HeteroData graph with spatially separated worlds.

    Args:
        data: HeteroData graph to visualise.
        title: Figure title.
        show_attrs: When True, annotate each node with its ``x`` feature values
            and each edge with its ``edge_attr`` values (raw floats, no names).
    """
    if not isinstance(data, HeteroData):
        raise TypeError("Expected HeteroData")

    world_attr = getattr(data.get("object", data[data.node_types[0]]), "world", None)
    num_worlds = (
        len(world_attr.unique())
        if world_attr is not None and len(world_attr) > 0
        else 1
    )
    fig, ax = plt.subplots(figsize=(max(10, num_worlds * 5), 8))
    _draw_into_axes(fig, ax, data, title, show_attrs)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise a world-batched graph from AxionDatasetGNN"
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Root directory of the processed dataset"
    )
    parser.add_argument("--idx", type=int, default=0, help="Index of the graph within the dataset")
    parser.add_argument(
        "--show_attrs",
        action="store_true",
        help="Annotate nodes and edges with their raw feature values",
    )
    args = parser.parse_args()

    dataset = AxionDatasetGNN(root=args.dataset_root)
    n_graphs = len(dataset)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_axes([0.02, 0.09, 0.96, 0.89])
    ax_prev = fig.add_axes([0.38, 0.01, 0.10, 0.05])
    ax_idx  = fig.add_axes([0.49, 0.01, 0.10, 0.05])
    ax_next = fig.add_axes([0.60, 0.01, 0.10, 0.05])

    btn_prev = Button(ax_prev, "◀ Prev")
    btn_next = Button(ax_next, "Next ▶")
    textbox = TextBox(ax_idx, "", initial=str(args.idx))

    state: dict = {"idx": args.idx, "drag_manager": None}

    def redraw(idx: int) -> None:
        if state["drag_manager"] is not None:
            state["drag_manager"].disconnect()
        state["idx"] = idx
        textbox.set_val(str(idx))
        dm = _draw_into_axes(
            fig, ax, dataset[idx],
            title=f"Dataset: {args.dataset_root}  |  idx={idx}",
            show_attrs=args.show_attrs,
        )
        state["drag_manager"] = dm
        fig.canvas.draw_idle()

    def on_prev(_event) -> None:
        redraw(max(0, state["idx"] - 1))

    def on_next(_event) -> None:
        redraw(min(n_graphs - 1, state["idx"] + 1))

    def on_submit(text: str) -> None:
        try:
            idx = int(text)
        except ValueError:
            textbox.set_val(str(state["idx"]))
            return
        idx = max(0, min(n_graphs - 1, idx))
        if idx != state["idx"]:
            redraw(idx)
        else:
            textbox.set_val(str(idx))

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    textbox.on_submit(on_submit)

    redraw(state["idx"])
    plt.show()
