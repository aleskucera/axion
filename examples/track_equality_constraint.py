import os
import pathlib
import tempfile
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

os.environ["PYOPENGL_PLATFORM"] = "glx"

CONFIG_PATH = pathlib.Path(__file__).parent.joinpath("conf")

EQ_TYPE_TRACK = 4


# --- Track Geometry Logic (Python side for Mesh Generation) ---
class Track2D:
    def __init__(self, r_rear, r_front, dist):
        self.r1 = r_rear
        self.r2 = r_front
        self.dist = dist
        self.c1 = np.array([0.0, 0.0])
        self.c2 = np.array([dist, 0.0])

        D_vec = self.c2 - self.c1
        L = np.linalg.norm(D_vec)
        val = np.clip((self.r1 - self.r2) / L, -1.0, 1.0)
        beta = np.arccos(val)

        dir_vec = np.array([1.0, 0.0])

        def rotate(v, angle):
            c, s = np.cos(angle), np.sin(angle)
            return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])

        n_top = rotate(dir_vec, beta)
        self.p1_top = self.c1 + self.r1 * n_top
        self.p2_top = self.c2 + self.r2 * n_top
        n_bot = rotate(dir_vec, -beta)
        self.p1_bot = self.c1 + self.r1 * n_bot
        self.p2_bot = self.c2 + self.r2 * n_bot

        self.len_top = np.linalg.norm(self.p2_top - self.p1_top)
        self.len_bot = np.linalg.norm(self.p1_bot - self.p2_bot)

        ang_f_start = np.arctan2(n_top[1], n_top[0])
        ang_f_end = np.arctan2(n_bot[1], n_bot[0])
        self.sweep_front = (ang_f_start - ang_f_end) % (2 * np.pi)
        self.len_arc_front = self.sweep_front * self.r2

        ang_r_start = np.arctan2(n_bot[1], n_bot[0])
        ang_r_end = np.arctan2(n_top[1], n_top[0])
        self.sweep_rear = (ang_r_start - ang_r_end) % (2 * np.pi)
        self.len_arc_rear = self.sweep_rear * self.r1

        self.ang_f_start = ang_f_start
        self.ang_r_start = ang_r_start

        self.L1 = self.len_top
        self.L2 = self.L1 + self.len_arc_front
        self.L3 = self.L2 + self.len_bot
        self.total_len = self.L3 + self.len_arc_rear

    def get_pos(self, u_in):
        u = u_in % self.total_len
        if u < self.L1:
            t = u / self.len_top
            return (1 - t) * self.p1_top + t * self.p2_top, 0
        elif u < self.L2:
            arc_u = u - self.L1
            angle = self.ang_f_start - (arc_u / self.r2)
            return self.c2 + self.r2 * np.array([np.cos(angle), np.sin(angle)]), 1
        elif u < self.L3:
            line_u = u - self.L2
            t = line_u / self.len_bot
            return (1 - t) * self.p2_bot + t * self.p1_bot, 0
        else:
            arc_u = u - self.L3
            angle = self.ang_r_start - (arc_u / self.r1)
            return self.c1 + self.r1 * np.array([np.cos(angle), np.sin(angle)]), 1


def generate_track_data(r1, r2, dist, tube_radius=0.1, segments=200, sides=12):
    track = Track2D(r1, r2, dist)

    vertices = []
    indices = []

    u_vals = np.linspace(0, track.total_len, segments, endpoint=False)

    for i, u in enumerate(u_vals):
        pos_2d, _ = track.get_pos(u)

        pos_next, _ = track.get_pos(u + 0.001)
        tangent = pos_next - pos_2d
        tangent /= np.linalg.norm(tangent)

        normal = np.array([-tangent[1], tangent[0]])  # 2D normal
        center = np.array([pos_2d[0], pos_2d[1], 0.0])

        for s in range(sides):
            theta = 2.0 * np.pi * s / sides
            off_n = np.cos(theta) * tube_radius
            off_b = np.sin(theta) * tube_radius

            vx = center[0] + off_n * normal[0]
            vy = center[1] + off_n * normal[1]
            vz = center[2] + off_b * 1.0

            vertices.append([vx, vy, vz])

    for i in range(segments):
        i_next = (i + 1) % segments
        base = i * sides
        base_next = i_next * sides

        for s in range(sides):
            s_next = (s + 1) % sides

            v1 = base + s
            v2 = base_next + s
            v3 = base_next + s_next
            v4 = base + s_next

            indices.append(v1)
            indices.append(v2)
            indices.append(v3)
            indices.append(v1)
            indices.append(v3)
            indices.append(v4)

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)


class Simulator(AbstractSimulator):
    def __init__(
        self,
        sim_config: SimulationConfig,
        render_config: RenderingConfig,
        exec_config: ExecutionConfig,
        engine_config: EngineConfig,
        logging_config: LoggingConfig,
    ):
        super().__init__(
            sim_config,
            render_config,
            exec_config,
            engine_config,
            logging_config,
        )

    def build_model(self) -> newton.Model:
        # 1. Track Parameters
        r_rear = 1.5
        r_front = 0.8
        dist = 5.0

        # Generate Visual Mesh
        verts, indices = generate_track_data(r_rear, r_front, dist, tube_radius=0.05)
        mesh_obj = newton.Mesh(vertices=verts, indices=indices)

        # Track Orientation (Parent Transform)
        q_tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5)
        p_track_origin = wp.vec3(0.0, 0.0, 1.0)
        X_p = wp.transform(p_track_origin, q_tilt)

        # 2. Base Body (Static) to attach visual
        # Note: Equality constraint can attach to body -1, but for visualizing the track
        # rigidly moving with a base, we can make a static base.
        # Or just attach visual to world. Let's make a static base.
        base = self.builder.add_body(key="base", mass=0.0)

        visual_cfg = newton.ModelBuilder.ShapeConfig(is_visible=True, has_shape_collision=False)
        self.builder.add_shape_mesh(
            body=base, mesh=mesh_obj, xform=X_p, cfg=visual_cfg, key="track_visual"
        )

        # 3. Rider (Dynamic Box)
        # Start it near the track
        rider = self.builder.add_body(
            key="rider", mass=1.0, xform=wp.transform((0.0, 0.0, 2.5), wp.quat_identity())
        )
        self.builder.add_shape_box(
            rider, hx=0.2, hy=0.1, hz=0.6, cfg=newton.ModelBuilder.ShapeConfig(density=100)
        )

        # We need a free joint for the rider to move (unless using Articulation)
        # Actually, if we use add_body (which assumes FREE joint implicitly in some contexts or needs articulation?),
        # in Newton/Axion, bodies need to be part of an articulation or just floating?
        # Standard Newton builder: add_body just adds body props.
        # We need to add a JointType.FREE to make it dynamic if it's the root of an articulation.
        # Let's use standard articulation flow.

        # Re-do Rider setup as link + free joint
        # Actually, add_body is low level.
        # add_link is better if building articulation.
        # But here we want independent bodies connected by equality constraint.
        # So we can have two separate articulations (one static, one free).

        # Base Articulation (Static)
        joint_base = self.builder.add_joint(
            joint_type=newton.JointType.FIXED, parent=-1, child=base, key="base_joint"
        )
        self.builder.add_articulation([joint_base], key="base_arti")

        # Rider Articulation (Free)
        joint_rider = self.builder.add_joint(
            joint_type=newton.JointType.FREE, parent=-1, child=rider, key="rider_joint"
        )
        self.builder.add_articulation([joint_rider], key="rider_arti")

        # 4. Track Equality Constraint
        # Map parameters to Anchor
        track_params = wp.vec3(dist, r_rear, r_front)

        # Relpose: The transform of the Track Frame relative to Body 1 (Base).
        # We put the visual mesh at X_p relative to Base.
        # So the Track Frame is X_p.
        track_relpose = X_p

        self.builder.add_equality_constraint(
            constraint_type=EQ_TYPE_TRACK,
            body1=base,
            body2=rider,
            anchor=track_params,
            relpose=track_relpose,
            key="track_eq",
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def track_equality_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    simulator = Simulator(
        sim_config=sim_config,
        render_config=render_config,
        exec_config=exec_config,
        engine_config=engine_config,
        logging_config=logging_config,
    )

    simulator.run()


if __name__ == "__main__":
    track_equality_example()
