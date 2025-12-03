from importlib.resources import files
from typing import override

import hydra
import newton
import numpy as np
import warp as wp
from axion.core.control_utils import JointMode
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

CONFIG_PATH = files("axion").joinpath("examples").joinpath("conf")

def update_ground_plane(
    builder,
    pos,
    rot,
    ke: float = None,
    kd: float = None,
    kf: float = None,
    mu: float = None,
    restitution: float = None,
):
    normal = Rotation.from_quat(rot).as_matrix() @ np.array([0., 1., 0.])
    d = np.dot(pos, normal)
    # print(normal, d)
    builder._ground_params = {
        'plane': [*normal, d],
        'pos': pos,
        'rot': rot,
        'width': 0.0,
        'length': 0.0,
        'ke': ke if ke is not None else builder.default_shape_ke,
        'kd': kd if kd is not None else builder.default_shape_kd,
        'kf': kf if kf is not None else builder.default_shape_kf,
        'mu': mu if mu is not None else builder.ShapeConfig.mu,
        'restitution': restitution if restitution is not None else builder.ShapeConfig.restitution
    }

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

    @override
    def control_policy(self, state: newton.State):
        wp.copy(self.control.joint_f, wp.array([0.0, 800.0], dtype=wp.float32))

    def build_model(self) -> newton.Model:
        self.builder.up_axis = newton.Axis.Y

        self.joint_type = newton.JointType.REVOLUTE
        self.chain_length = 2
        self.chain_width = 1.5

        self.lower = -2 * wp.pi
        self.upper = 2 * wp.pi
        self.limitd_ke = 0.0
        self.limitd_kd = 0.0

        shape_ke = 1.0e4
        shape_kd = 1.0e3
        shape_kf = 1.0e4

        ground_config = newton.ModelBuilder.ShapeConfig(
            ke = shape_ke,
            kd = shape_kd,
            kf = shape_kf
        )

        self.builder.add_ground_plane(
            cfg= ground_config
        )

        # contact configuration
        CONTACT_CONFIG = 0 # 0: no contact, 1-6: contact with ground

        for i in range(self.chain_length):
            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 1.0, 2.0], wp.quat_identity())
            else:
                parent = self.builder.joint_count - 1
                parent_joint_xform = wp.transform(
                    [self.chain_width, 0.0, 0.0], wp.quat_identity()
                )

            b = self.builder.add_body(
                xform=wp.transform( p = wp.vec3(i, 1.0, 0.0), q= wp.quat_identity()),
                armature= 0.1,
            )

            shape_offset = wp.transform(
                p=wp.vec3(self.chain_width * 0.5, 0.0, 0.0),  # old pos
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.radians(90.0))  # rotate Y→X to match old up_axis=0
            )

            # create shape
            link_config = newton.ModelBuilder.ShapeConfig(density=500.0, ke = shape_ke, kd = shape_kd, kf = shape_kf)
            self.builder.add_shape_capsule(
                body= b,
                xform= shape_offset,
                half_height= self.chain_width * 0.5,
                radius= 0.1,
                cfg= link_config
            )

            # # NERD (WP.SIM) IMPLEMENTATION:
            # # create shape
            # self.builder.add_shape_capsule(
            #     pos=(self.chain_width * 0.5, 0.0, 0.0),
            #     half_height=self.chain_width * 0.5,
            #     radius=0.1,
            #     up_axis=0,
            #     density=500.0,
            #     body=b,
            #     ke=shape_ke,
            #     kd=shape_kd,
            #     kf=shape_kf,
            # )

            self.builder.add_joint_revolute(
                parent=parent,
                child=b,
                axis=(0.0, 1.0, 0.0),
                parent_xform=parent_joint_xform,
                limit_lower=self.lower,
                limit_upper=self.upper,
                limit_ke=self.limitd_ke,
                limit_kd=self.limitd_kd,
            )

        self.builder.joint_q[:] = [0.0, 0.0]

        # 7 contact configurations used in the paper
        if CONTACT_CONFIG == 0:
            # contact-free
            offset = -15.5
            rot_xyz = np.array([0., 0., 0.])
        elif CONTACT_CONFIG == 1:
            # config 1
            offset = 0.0
            rot_xyz = np.array([0., 0., 0.])
        elif CONTACT_CONFIG == 2:
            # config 2
            offset = 0.2
            rot_xyz = np.array([wp.pi / 8., wp.pi / 16, wp.pi / 16.])
        elif CONTACT_CONFIG == 3:
            # config 3
            offset = 0.5
            rot_xyz = np.array([0., 0., 0.])
        elif CONTACT_CONFIG == 4:
            # config 4
            offset = -0.5
            rot_xyz = np.array([0., 0., 0.])
        elif CONTACT_CONFIG == 5:
            # config 5
            offset = -0.3
            rot_xyz = np.array([0., 0., 0.])
        elif CONTACT_CONFIG == 6:
            # config 6
            offset = 0.0
            rot_xyz = np.array([wp.pi / 8., 0., 0.])
        else:
            raise ValueError(f"Invalid contact configuration: {CONTACT_CONFIG}")
        
        rot = Rotation.from_euler('xyz', rot_xyz).as_quat()
        update_ground_plane(
            self.builder,
            pos=[0.0, offset, 0.0],
            rot=rot,
            ke=shape_ke,
            kd=shape_kd,
            kf=shape_kf,
        )

        model = self.builder.finalize()
        return model


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def basic_pendulum_example(cfg: DictConfig):
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
    basic_pendulum_example()
