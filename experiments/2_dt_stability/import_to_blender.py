"""Import a 2_dt_stability MuJoCo trajectory.npz as a Blender scene.

Each "iteration" in the npz corresponds to one ``dt`` value. Iterations are
stacked on the timeline so the same obstacle traversal plays back-to-back at
progressively coarser timesteps. A floating header above the chassis names the
current ``dt``. A fading breadcrumb trail shows the chassis path; older
iterations' trails fade out exponentially.

No ghost / target robot — this experiment has no comparison reference.

Run inside Blender:
    blender --background --python experiments/2_dt_stability/import_to_blender.py -- TRAJ.npz
    blender --python experiments/2_dt_stability/import_to_blender.py -- TRAJ.npz --output animated.blend

The npz is produced by experiments/2_dt_stability/sweep_mujoco_blender.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion
from mathutils import Vector

# Newton / MuJoCo geom type ints (the two enums share these values for our subset).
GEO_PLANE = 0
GEO_SPHERE = 2
GEO_CAPSULE = 3
GEO_ELLIPSOID = 4
GEO_CYLINDER = 5
GEO_BOX = 6
GEO_MESH = 7
GEO_CONE = 9
GEO_CONVEX_MESH = 10

# Vibrant per-body palette for the live robot.
LIVE_PALETTE = {
    0: (0.05, 0.45, 1.00),  # chassis
    1: (0.95, 0.15, 0.55),  # body 1 (e.g. left wheel): magenta
    2: (0.10, 0.85, 0.85),  # body 2 (e.g. right wheel): cyan
    3: (0.85, 0.95, 0.10),  # body 3 (e.g. rear wheel): lime
}
LIVE_FALLBACK = (0.85, 0.45, 0.10)
STATIC_COLOR = (0.22, 0.22, 0.24)  # ground plane: muted gray
OBSTACLE_COLOR = (0.95, 0.70, 0.15)  # static non-plane shapes: warm yellow-orange

CAMERA_BBOX_MAX_EXTENT = 15.0  # clip diverging chassis positions out of the bbox


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("npz", type=Path, help="trajectory.npz from sweep_mujoco_blender.py")
    p.add_argument("--output", type=Path, default=None, help="Save to this .blend after import")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Materials and shape primitives
# ---------------------------------------------------------------------------

def make_material(
    name: str,
    color: tuple[float, float, float],
    alpha: float,
    roughness: float | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Alpha"].default_value = alpha
    if roughness is not None and "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = float(roughness)
    if emission_strength > 0.0:
        for key in ("Emission Color", "Emission"):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = (*color, 1.0)
                break
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = float(emission_strength)
    if alpha < 1.0:
        for attr, value in (("blend_method", "HASHED"), ("shadow_method", "HASHED")):
            if hasattr(mat, attr):
                setattr(mat, attr, value)
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "DITHERED"
    return mat


def live_color_for_body(body_idx: int) -> tuple[float, float, float]:
    if body_idx == -1:
        return STATIC_COLOR
    return LIVE_PALETTE.get(body_idx, LIVE_FALLBACK)


def make_shape_object(shape: dict, name: str) -> bpy.types.Object:
    """Create a Blender mesh object whose local geometry matches the shape descriptor."""
    gt = int(shape["geo_type"])
    scale = np.asarray(shape["geo_scale"], dtype=np.float32)

    if gt == GEO_BOX:
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        obj = bpy.context.active_object
        obj.scale = (float(scale[0]), float(scale[1]), float(scale[2]))
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif gt == GEO_SPHERE:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=float(scale[0]))
        obj = bpy.context.active_object
    elif gt in (GEO_CYLINDER, GEO_CAPSULE):
        bpy.ops.mesh.primitive_cylinder_add(radius=float(scale[0]), depth=2.0 * float(scale[1]))
        obj = bpy.context.active_object
    elif gt == GEO_CONE:
        bpy.ops.mesh.primitive_cone_add(radius1=float(scale[0]), depth=2.0 * float(scale[1]))
        obj = bpy.context.active_object
    elif gt in (GEO_MESH, GEO_CONVEX_MESH):
        verts = np.asarray(shape["mesh_verts"], dtype=np.float32) * scale
        faces = np.asarray(shape["mesh_faces"], dtype=np.int32)
        mesh = bpy.data.meshes.new(name=f"{name}_mesh")
        mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
        mesh.update()
        obj = bpy.data.objects.new(name=name, object_data=mesh)
        bpy.context.collection.objects.link(obj)
    elif gt == GEO_PLANE:
        sx = float(scale[0]) if scale[0] > 0 else 50.0
        sy = float(scale[1]) if scale[1] > 0 else 50.0
        bpy.ops.mesh.primitive_plane_add(size=2.0)
        obj = bpy.context.active_object
        obj.scale = (sx, sy, 1.0)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    else:
        raise ValueError(f"Unsupported geo_type {gt} for shape {name}")

    obj.name = name
    return obj


def warp_xform_to_blender(xf7: np.ndarray) -> tuple[Vector, Quaternion]:
    """[tx, ty, tz, qx, qy, qz, qw] → (location, Quaternion(w, x, y, z))."""
    return (
        Vector((float(xf7[0]), float(xf7[1]), float(xf7[2]))),
        Quaternion((float(xf7[6]), float(xf7[3]), float(xf7[4]), float(xf7[5]))),
    )


# ---------------------------------------------------------------------------
# Robot construction
# ---------------------------------------------------------------------------

def build_robot(
    shapes: list[dict],
    namespace: str,
    material_for_shape,
    static_collection: bpy.types.Collection,
    body_collection: bpy.types.Collection,
) -> dict[int, bpy.types.Object]:
    """Per-body Empties parented to world; each shape is a child mesh.

    ``material_for_shape(shape: dict) -> bpy.types.Material`` lets the caller
    pick a material per shape (so e.g. a static box obstacle and a static
    ground plane can have different materials).
    """
    body_empties: dict[int, bpy.types.Object] = {}
    for i, shape in enumerate(shapes):
        body_idx = int(shape["body_idx"])
        local_xf = np.asarray(shape["local_xform"], dtype=np.float32)
        loc, quat = warp_xform_to_blender(local_xf)

        obj_name = f"{namespace}_shape_{i}"
        obj = make_shape_object(shape, obj_name)
        obj.data.materials.clear()
        obj.data.materials.append(material_for_shape(shape))

        if body_idx == -1:
            obj.location = loc
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = quat
            static_collection.objects.link(obj)
            try:
                bpy.context.collection.objects.unlink(obj)
            except RuntimeError:
                pass
            continue

        if body_idx not in body_empties:
            empty = bpy.data.objects.new(f"{namespace}_body_{body_idx}", None)
            empty.empty_display_type = "ARROWS"
            empty.empty_display_size = 0.1
            body_collection.objects.link(empty)
            body_empties[body_idx] = empty

        body = body_empties[body_idx]
        obj.parent = body
        obj.location = loc
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = quat
        body_collection.objects.link(obj)
        try:
            bpy.context.collection.objects.unlink(obj)
        except RuntimeError:
            pass

    return body_empties


def keyframe_bodies(
    body_empties: dict[int, bpy.types.Object],
    poses: np.ndarray,  # [T, num_bodies, 7]
    frame_offset: int = 0,
):
    T = poses.shape[0]
    for body_idx, empty in body_empties.items():
        empty.rotation_mode = "QUATERNION"
        for t in range(T):
            xf = poses[t, body_idx]
            loc, quat = warp_xform_to_blender(xf)
            empty.location = loc
            empty.rotation_quaternion = quat
            f = frame_offset + t
            empty.keyframe_insert("location", frame=f)
            empty.keyframe_insert("rotation_quaternion", frame=f)


# ---------------------------------------------------------------------------
# Lighting and camera
# ---------------------------------------------------------------------------

def clear_default_objects():
    for name in ("Cube",):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)


def setup_lighting(scene: bpy.types.Scene):
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    light_data = bpy.data.lights.new(name="key_sun", type="SUN")
    light_data.energy = 4.0
    light_data.color = (1.0, 0.92, 0.78)
    if hasattr(light_data, "angle"):
        light_data.angle = np.radians(3.0)
    light_obj = bpy.data.objects.new(name="key_sun", object_data=light_data)
    light_obj.rotation_euler = (np.radians(55), np.radians(20), np.radians(35))
    scene.collection.objects.link(light_obj)

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is not None:
        bg_node.inputs["Color"].default_value = (0.04, 0.05, 0.07, 1.0)
        bg_node.inputs["Strength"].default_value = 0.3


def setup_camera(scene: bpy.types.Scene, points: np.ndarray):
    """Static elevated 3/4 camera framing the bounding box of ``points`` ([N, 3])."""
    for obj in list(scene.objects):
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = (pmin + pmax) / 2.0
    extent = float(np.linalg.norm(pmax - pmin))

    aim = bpy.data.objects.new("camera_aim", None)
    aim.empty_display_type = "PLAIN_AXES"
    aim.empty_display_size = 0.1
    aim.location = (float(center[0]), float(center[1]), float(center[2]))
    scene.collection.objects.link(aim)

    cam_data = bpy.data.cameras.new("camera")
    cam_data.lens = 50.0
    cam_obj = bpy.data.objects.new("camera", cam_data)

    azimuth = np.radians(35.0)
    elevation = np.radians(40.0)
    distance = max(extent * 0.95, 6.0)
    offset = np.array(
        [
            distance * np.cos(elevation) * np.sin(azimuth),
            -distance * np.cos(elevation) * np.cos(azimuth),
            distance * np.sin(elevation),
        ],
        dtype=np.float32,
    )
    cam_obj.location = (
        float(center[0] + offset[0]),
        float(center[1] + offset[1]),
        float(center[2] + offset[2]),
    )

    track = cam_obj.constraints.new("TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj


# ---------------------------------------------------------------------------
# Floating labels
# ---------------------------------------------------------------------------

class _ConstantInterpolation:
    """Force CONSTANT interpolation on keyframes inserted inside the block."""

    def __enter__(self):
        edit = bpy.context.preferences.edit
        self._prev = edit.keyframe_new_interpolation_type
        edit.keyframe_new_interpolation_type = "CONSTANT"
        return self

    def __exit__(self, exc_type, exc, tb):
        bpy.context.preferences.edit.keyframe_new_interpolation_type = self._prev


def make_floating_label(
    name: str,
    follow_body: bpy.types.Object,
    text: str,
    size: float,
    z_offset: float,
    collection: bpy.types.Collection,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    camera: bpy.types.Object | None = None,
) -> bpy.types.Object:
    """Text that follows ``follow_body``'s position; faces the camera if one exists."""
    curve = bpy.data.curves.new(name=f"{name}_curve", type="FONT")
    curve.body = text
    curve.size = size
    curve.align_x = "CENTER"
    curve.align_y = "BOTTOM"
    obj = bpy.data.objects.new(name=name, object_data=curve)

    mat = make_material(f"{name}_mat", color, 1.0)
    obj.data.materials.append(mat)

    obj.location = (0.0, 0.0, z_offset)
    copy_loc = obj.constraints.new("COPY_LOCATION")
    copy_loc.target = follow_body
    copy_loc.use_offset = True

    if camera is not None:
        track = obj.constraints.new("TRACK_TO")
        track.target = camera
        track.track_axis = "TRACK_Z"
        track.up_axis = "UP_Y"

    collection.objects.link(obj)
    return obj


def make_per_iteration_labels(
    follow_body: bpy.types.Object,
    iter_labels: list[str],
    T: int,
    collection: bpy.types.Collection,
    camera: bpy.types.Object | None,
    size: float = 0.3,
    z_offset: float = 0.6,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """One floating label per iteration above ``follow_body``; keyframed visible only during its segment."""
    n_iters = len(iter_labels)
    with _ConstantInterpolation():
        for it_idx, text in enumerate(iter_labels):
            obj = make_floating_label(
                name=f"label_iter_{it_idx}",
                follow_body=follow_body,
                text=text,
                size=size,
                z_offset=z_offset,
                collection=collection,
                color=color,
                camera=camera,
            )
            start = it_idx * T
            end = (it_idx + 1) * T
            obj.hide_render = True
            obj.hide_viewport = True
            obj.keyframe_insert("hide_render", frame=0)
            obj.keyframe_insert("hide_viewport", frame=0)
            obj.hide_render = False
            obj.hide_viewport = False
            obj.keyframe_insert("hide_render", frame=start)
            obj.keyframe_insert("hide_viewport", frame=start)
            if it_idx < n_iters - 1:
                obj.hide_render = True
                obj.hide_viewport = True
                obj.keyframe_insert("hide_render", frame=end)
                obj.keyframe_insert("hide_viewport", frame=end)


# ---------------------------------------------------------------------------
# Breadcrumb trail (chassis path tagged with short equidistant dashes)
# ---------------------------------------------------------------------------

_BOX_VERTS = np.array(
    [
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)
_BOX_FACES = [
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
    (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3),
]


def _rot_from_x_to(t: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(t))
    if n < 1e-9:
        return np.eye(3, dtype=np.float32)
    t = (t / n).astype(np.float32)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cosA = float(np.dot(x, t))
    if cosA > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float32)
    if cosA < -1.0 + 1e-6:
        return np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    axis = np.cross(x, t)
    sinA = float(np.linalg.norm(axis))
    axis = axis / sinA
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + sinA * K + (1.0 - cosA) * (K @ K)).astype(np.float32)


def _tangents(points: np.ndarray) -> np.ndarray:
    t = np.empty_like(points)
    if len(points) == 1:
        t[0] = (1.0, 0.0, 0.0)
        return t
    t[0] = points[1] - points[0]
    t[-1] = points[-1] - points[-2]
    if len(points) > 2:
        t[1:-1] = points[2:] - points[:-2]
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return (t / norms).astype(np.float32)


def _build_segment_cluster(
    name: str,
    centers: np.ndarray,
    tangents: np.ndarray,
    length: float,
    width: float,
):
    n_base = len(_BOX_VERTS)
    scale = np.array([length, width, width], dtype=np.float32)
    all_verts = np.empty((len(centers) * n_base, 3), dtype=np.float32)
    all_faces: list[tuple[int, ...]] = []
    for i, (c, t) in enumerate(zip(centers, tangents)):
        rot = _rot_from_x_to(t)
        all_verts[i * n_base : (i + 1) * n_base] = (_BOX_VERTS * scale) @ rot.T + c
        offset = i * n_base
        for f in _BOX_FACES:
            all_faces.append(tuple(int(idx + offset) for idx in f))
    mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    mesh.from_pydata(all_verts.tolist(), [], all_faces)
    mesh.update()
    return bpy.data.objects.new(name=name, object_data=mesh)


def _resample_equidistant(path: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.diff(path, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cumlen[-1])
    if total < 1e-6:
        return path[:1].astype(np.float32), np.zeros(1, dtype=np.float32)
    n = max(2, int(round(total / spacing)) + 1)
    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 3), dtype=np.float32)
    for i, s in enumerate(targets):
        idx = int(np.searchsorted(cumlen, s, side="right")) - 1
        idx = max(0, min(idx, len(seg_len) - 1))
        denom = cumlen[idx + 1] - cumlen[idx]
        t = 0.0 if denom < 1e-12 else (s - cumlen[idx]) / denom
        out[i] = path[idx] + t * (path[idx + 1] - path[idx])
    return out, targets.astype(np.float32)


def _spawn_steps(path: np.ndarray, arc_lengths: np.ndarray) -> np.ndarray:
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    return np.searchsorted(cumlen, arc_lengths, side="left").astype(np.int32)


def make_breadcrumb_trails(
    body_pose_iters: np.ndarray,  # [N_iters, T, num_bodies, 7]
    T: int,
    color: tuple[float, float, float],
    collection: bpy.types.Collection,
    spacing: float = 0.4,
    segment_length: float = 0.18,
    segment_width: float = 0.04,
    decay_rate: float = 0.4,
    min_alpha: float = 0.05,
):
    """Equidistant chassis-path dashes per iteration; spawn as the chassis crosses each location."""
    n_iters = body_pose_iters.shape[0]
    for it_idx in range(n_iters):
        path = body_pose_iters[it_idx, :, 0, :3].astype(np.float32)
        sampled, arc_lengths = _resample_equidistant(path, spacing)
        tangents = _tangents(sampled)
        spawn_steps = _spawn_steps(path, arc_lengths)

        mat = make_material(f"breadcrumb_{it_idx}", color, 0.99, emission_strength=1.5)
        iter_start = it_idx * T

        for k in range(len(sampled)):
            seg = _build_segment_cluster(
                f"breadcrumb_{it_idx}_seg_{k}",
                sampled[k : k + 1],
                tangents[k : k + 1],
                length=segment_length,
                width=segment_width,
            )
            seg.data.materials.append(mat)
            collection.objects.link(seg)

            spawn_frame = iter_start + int(spawn_steps[k])
            with _ConstantInterpolation():
                seg.hide_render = True
                seg.hide_viewport = True
                seg.keyframe_insert("hide_render", frame=0)
                seg.keyframe_insert("hide_viewport", frame=0)
                seg.hide_render = False
                seg.hide_viewport = False
                seg.keyframe_insert("hide_render", frame=spawn_frame)
                seg.keyframe_insert("hide_viewport", frame=spawn_frame)

        alpha_input = mat.node_tree.nodes["Principled BSDF"].inputs["Alpha"]
        alpha_input.default_value = 1.0
        alpha_input.keyframe_insert("default_value", frame=iter_start)
        alpha_input.default_value = 1.0
        alpha_input.keyframe_insert("default_value", frame=iter_start + T)
        for later in range(it_idx + 1, n_iters):
            age = later - it_idx
            alpha = min_alpha + (1.0 - min_alpha) * (decay_rate**age)
            alpha_input.default_value = alpha
            alpha_input.keyframe_insert("default_value", frame=(later + 1) * T)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    clear_default_objects()
    data = np.load(args.npz, allow_pickle=True)
    fps = float(data["fps"])
    body_pose_iters = np.asarray(data["body_pose_iters"])  # [N_iters, T, num_bodies, 7]
    iter_labels = [str(x) for x in data["iter_labels"]]
    iter_stable = (
        np.asarray(data["iter_stable"], dtype=bool)
        if "iter_stable" in data.files
        else np.ones(body_pose_iters.shape[0], dtype=bool)
    )
    shapes = list(data["shapes"])

    n_iters = body_pose_iters.shape[0]
    T = body_pose_iters.shape[1]
    T_total = n_iters * T
    print(
        f"Loaded {args.npz}: {n_iters} iterations × {T} frames = {T_total} frames @ {fps:.1f} fps; "
        f"{len(shapes)} shapes; stable mask = {iter_stable.tolist()}"
    )

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = max(0, T_total - 1)
    scene.render.fps = max(1, int(round(fps)))

    setup_lighting(scene)

    # Camera framing: chassis positions across stable iterations only, clipped to
    # CAMERA_BBOX_MAX_EXTENT so divergent runs don't blow out the framing.
    chassis_points = body_pose_iters[iter_stable, :, 0, :3].reshape(-1, 3)
    if len(chassis_points) == 0:
        chassis_points = body_pose_iters[:, :, 0, :3].reshape(-1, 3)
    chassis_points = chassis_points[
        np.all(np.abs(chassis_points) < CAMERA_BBOX_MAX_EXTENT, axis=1)
    ]
    if len(chassis_points) == 0:
        chassis_points = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.5]], dtype=np.float32
        )
    setup_camera(scene, chassis_points.astype(np.float32))

    static_coll = bpy.data.collections.new("static")
    live_coll = bpy.data.collections.new("live")
    for c in (static_coll, live_coll):
        scene.collection.children.link(c)

    live_materials: dict[str, bpy.types.Material] = {}

    def material_for_live(shape: dict) -> bpy.types.Material:
        body_idx = int(shape["body_idx"])
        geo_type = int(shape["geo_type"])
        if body_idx == -1:
            if geo_type == GEO_PLANE:
                key = "ground"
                if key not in live_materials:
                    live_materials[key] = make_material(
                        "ground", STATIC_COLOR, 1.0, roughness=0.95
                    )
            else:
                key = "obstacle"
                if key not in live_materials:
                    live_materials[key] = make_material(
                        "obstacle", OBSTACLE_COLOR, 1.0, roughness=0.5, emission_strength=0.4
                    )
            return live_materials[key]

        key = f"body_{body_idx}"
        if key not in live_materials:
            live_materials[key] = make_material(
                f"live_body_{body_idx}", live_color_for_body(body_idx), 1.0
            )
        return live_materials[key]

    live_bodies = build_robot(shapes, "live", material_for_live, static_coll, live_coll)
    for empty in live_bodies.values():
        empty.empty_display_size = 0.0

    label_coll = bpy.data.collections.new("floating_labels")
    scene.collection.children.link(label_coll)
    camera = bpy.context.scene.camera
    if 0 in live_bodies:
        make_per_iteration_labels(
            follow_body=live_bodies[0],
            iter_labels=iter_labels,
            T=T,
            collection=label_coll,
            camera=camera,
            color=LIVE_PALETTE[0],
        )

    trail_coll = bpy.data.collections.new("breadcrumbs")
    scene.collection.children.link(trail_coll)
    make_breadcrumb_trails(
        body_pose_iters,
        T=T,
        color=LIVE_PALETTE[0],
        collection=trail_coll,
    )

    for it in range(n_iters):
        offset = it * T
        keyframe_bodies(live_bodies, body_pose_iters[it], frame_offset=offset)

    if args.output:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
