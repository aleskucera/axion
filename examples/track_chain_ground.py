import os
import pathlib
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

    def get_frame(self, u_in):
        u = u_in % self.total_len
        pos = np.zeros(2)
        tan = np.zeros(2)

        if u < self.L1:
            t = u / self.len_top
            pos = (1 - t) * self.p1_top + t * self.p2_top
            tan = (self.p2_top - self.p1_top) / self.len_top
        elif u < self.L2:
            arc_u = u - self.L1
            angle = self.ang_f_start - (arc_u / self.r2)
            pos = self.c2 + self.r2 * np.array([np.cos(angle), np.sin(angle)])
            tan = np.array([np.sin(angle), -np.cos(angle)])
        elif u < self.L3:
            line_u = u - self.L2
            t = line_u / self.len_bot
            pos = (1 - t) * self.p2_bot + t * self.p1_bot
            tan = (self.p1_bot - self.p2_bot) / self.len_bot
        else:
            arc_u = u - self.L3
            angle = self.ang_r_start - (arc_u / self.r1)
            pos = self.c1 + self.r1 * np.array([np.cos(angle), np.sin(angle)])
            tan = np.array([np.sin(angle), -np.cos(angle)])

        return pos, tan


def generate_track_data(r1, r2, dist, tube_radius=0.1, segments=200, sides=12):
    track = Track2D(r1, r2, dist)
    vertices = []
    indices = []
    u_vals = np.linspace(0, track.total_len, segments, endpoint=False)

    for i, u in enumerate(u_vals):
        pos_2d, tan_2d = track.get_frame(u)
        tangent = np.array([tan_2d[0], tan_2d[1], 0.0])
        normal = np.array([-tan_2d[1], tan_2d[0], 0.0])
        binormal = np.array([0.0, 0.0, 1.0])
        center = np.array([pos_2d[0], pos_2d[1], 0.0])

        for s in range(sides):
            theta = 2.0 * np.pi * s / sides
            off_n = np.cos(theta) * tube_radius
            off_b = np.sin(theta) * tube_radius
            pt = center + off_n * normal + off_b * binormal
            vertices.append(pt)

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
    def __init__(self, sim_config, render_config, exec_config, engine_config, logging_config):
        super().__init__(sim_config, render_config, exec_config, engine_config, logging_config)

    def build_model(self) -> newton.Model:
        # 1. Ground Plane
        self.builder.add_ground_plane()

        # 2. Track Parameters
        r_rear = 1.5
        r_front = 0.8
        dist = 5.0
        track_helper = Track2D(r_rear, r_front, dist)

        # Generate Visual Mesh
        verts, indices = generate_track_data(r_rear, r_front, dist, tube_radius=0.05)
        mesh_obj = newton.Mesh(vertices=verts, indices=indices)

        # Track Orientation
        # Tilt slightly and position it so links hit the ground
        q_tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.2)
        # Position track origin at Z=0.4. Since boxes have hz=0.45, they will drag.
        p_track_origin = wp.vec3(0.0, 0.0, 0.4)
        X_track = wp.transform(p_track_origin, q_tilt)

        # Static Base
        base = self.builder.add_body(key="base", mass=0.0)
        # base_joint = self.builder.add_joint(newton.JointType.FIXED, -1, base)
        # self.builder.add_articulation([base_joint])

        visual_cfg = newton.ModelBuilder.ShapeConfig(is_visible=True, has_shape_collision=False)
        self.builder.add_shape_mesh(
            body=base, mesh=mesh_obj, xform=X_track, cfg=visual_cfg, key="track_visual"
        )

        # 3. Chain Parameters (From your latest edits)
        num_links = 20
        link_spacing = 0.65
        start_u = 0.5
        hx, hy, hz = 0.3, 0.1, 0.45

        # 4. Create Chain Articulation
        # Initial Transform for Head
        pos_2d, tan_2d = track_helper.get_frame(start_u)
        pos_local = np.array([pos_2d[0], pos_2d[1], 0.0])
        tangent = np.array([tan_2d[0], tan_2d[1], 0.0])
        normal = np.array([-tan_2d[1], tan_2d[0], 0.0])
        binormal = np.array([0.0, 0.0, 1.0])
        q_local = wp.quat_from_matrix(
            wp.matrix_from_cols(wp.vec3(tangent), wp.vec3(normal), wp.vec3(binormal))
        )
        X_head_world = wp.transform_multiply(X_track, wp.transform(wp.vec3(pos_local), q_local))

        # Head Link
        head = self.builder.add_link(key="head", mass=0.0, xform=X_head_world)
        self.builder.add_shape_box(
            head,
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=newton.ModelBuilder.ShapeConfig(
                is_visible=True,
                has_shape_collision=True,
                density=100,
            ),
        )
        head_joint = self.builder.add_joint(newton.JointType.FREE, -1, head)

        all_joints = [head_joint]

        # Add Track Constraint for Head
        self.builder.add_equality_constraint(
            constraint_type=EQ_TYPE_TRACK,
            body1=base,
            body2=head,
            anchor=wp.vec3(dist, r_rear, r_front),
            relpose=X_track,
        )

        prev_link = head
        for i in range(1, num_links):
            u = start_u + i * link_spacing
            pos_2d, tan_2d = track_helper.get_frame(u)
            pos_local = np.array([pos_2d[0], pos_2d[1], 0.0])
            tangent = np.array([tan_2d[0], tan_2d[1], 0.0])
            normal = np.array([-tan_2d[1], tan_2d[0], 0.0])
            binormal = np.array([0.0, 0.0, 1.0])
            q_local = wp.quat_from_matrix(
                wp.matrix_from_cols(wp.vec3(tangent), wp.vec3(normal), wp.vec3(binormal))
            )
            X_curr_world = wp.transform_multiply(X_track, wp.transform(wp.vec3(pos_local), q_local))

            curr_link = self.builder.Pleaseadd_link(key=f"link_{i}", mass=0.0, xform=X_curr_world)
            self.builder.add_shape_box(
                curr_link,
                hx=hx,
                hy=hy,
                hz=hz,
                cfg=newton.ModelBuilder.ShapeConfig(
                    is_visible=True,
                    has_shape_collision=True,
                    density=100,
                ),
            )

            # Ball Joint connecting Prev -> Curr
            X_p_joint = wp.transform(wp.vec3(link_spacing / 2.0, 0.0, 0.0), wp.quat_identity())
            X_c_joint = wp.transform(wp.vec3(-link_spacing / 2.0, 0.0, 0.0), wp.quat_identity())

            ball_joint = self.builder.add_joint(
                newton.JointType.BALL,
                prev_link,
                curr_link,
                parent_xform=X_p_joint,
                child_xform=X_c_joint,
                custom_attributes={"joint_compliance": 1e-6},  # Stiff chain
            )
            all_joints.append(ball_joint)

            # Add Track Constraint for this link too
            self.builder.add_equality_constraint(
                constraint_type=EQ_TYPE_TRACK,
                body1=base,
                body2=curr_link,
                anchor=wp.vec3(dist, r_rear, r_front),
                relpose=X_track,
            )
            prev_link = curr_link

        self.builder.add_articulation(all_joints, key="chain_arti")

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="config", version_base=None)
def track_chain_ground_example(cfg: DictConfig):
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    simulator = Simulator(sim_config, render_config, exec_config, engine_config, logging_config)
    simulator.run()


if __name__ == "__main__":
    track_chain_ground_example()
