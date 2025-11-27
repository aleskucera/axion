from importlib.resources import files
from typing import override

import hydra
import newton
import warp as wp
# -------monkey-patch-adapter-approach-----
from axion.adapters import sim_adapter
wp.sim = sim_adapter
#-----------------------------------------
from axion.core.control_utils import JointMode
from axion import AbstractSimulator
from axion import EngineConfig
from axion import ExecutionConfig
from axion import LoggingConfig
from axion import RenderingConfig
from axion import SimulationConfig
from omegaconf import DictConfig

import torch
import yaml
import numpy as np
#from third_party.nerd.envs.neural_environment import NeruralEnvironment
from nerd.envs.neural_environment import NeuralEnvironment

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

CONFIG_PATH = files("axion").joinpath("examples").joinpath("conf")


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


    @override
    def init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ):
        # self.mujoco_solver.step(current_state, next_state, self.model.control(), contacts, dt)
        self.solver.integrate_bodies(self.model, current_state, next_state, dt)

    def build_model(self) -> newton.Model:
        chain_width = 1.5
        shape_ke = 1.0e4
        shape_kd = 1.0e3
        shape_kf = 1.0e4
    
        hx = chain_width*0.5

        link_0 = self.builder.add_body(armature=0.1)
        link_config = newton.ModelBuilder.ShapeConfig(density=500.0, ke = shape_ke, kd = shape_kd, kf = shape_kf)
        capsule_shape_transform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi/2))
        self.builder.add_shape_capsule(link_0,
                                       xform= capsule_shape_transform,
                                       radius=0.1, 
                                       half_height=chain_width*0.5,
                                       cfg = link_config)

        link_1 = self.builder.add_body(armature=0.1)
        self.builder.add_shape_capsule(link_1,
                                    xform = capsule_shape_transform,
                                    radius=0.1, 
                                    half_height=chain_width*0.5,
                                    cfg = link_config)

        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        self.builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 5.0), q=rot),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=1000.0,
            target_kd=50.0,
            custom_attributes={
                "joint_target_ki": [0.5],
                "joint_dof_mode": [JointMode.NONE],
            },
        )
        self.builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=500.0,
            target_kd=5.0,
            custom_attributes={
                "joint_target_ki": [0.5],
                "joint_dof_mode": [JointMode.NONE],
            },
            armature=0.1,
        )

        self.builder.add_ground_plane()

        model = self.builder.finalize()
        return model

def initialize_nerd():
    pass

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
