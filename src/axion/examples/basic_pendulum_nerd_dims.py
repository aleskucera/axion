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

# NeRD imports:
import torch
import yaml
import numpy as np
from pathlib import Path
#from third_party.nerd.envs.neural_environment import NeruralEnvironment
from nerd.envs.neural_environment import NeuralEnvironment
from nerd.utils.torch_utils import num_params_torch_model
# Repository base directory (project root)
# `parents[3]` from this file resolves to the repository root: /<repo>/
base_dir = Path(__file__).resolve().parents[3]
print(base_dir)

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
    """
    Initializes the neural integrator from NeRD.
    Uses the PendulumWithContactEnvironment specifically.
    """
    # Configuration
    device = 'cuda:0'
    model_path = base_dir /'third_party'/ 'nerd'/ 'nerd' /'pretrained_models' / 'NeRD_models' / 'Pendulum' / 'model' / 'nn' / 'model.pt'
    num_envs = 1
    num_steps = 5000
    seed = 42
    
    #set_random_seed(seed)
    
    # Load pretrained NeRD model
    print("Loading pretrained NeRD model...")
    neural_model, robot_name = torch.load(str(model_path), map_location=device, weights_only=False)
    print(f'Number of Model Parameters: {num_params_torch_model(neural_model)}')
    neural_model.to(device)
    
    # Load model configuration
    train_dir = (model_path.parent.parent).resolve()
    cfg_path = train_dir / 'cfg.yaml'
    print(cfg_path)
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    neural_integrator_cfg = cfg["env"]["neural_integrator_cfg"]

    env_cfg = {
        "env_name": "PendulumWithContact",
        "num_envs": num_envs,
        "render": False,  # Enable visualization
        "warp_env_cfg": {
            "seed": seed
        },
        "neural_integrator_cfg": neural_integrator_cfg,
        "neural_model": neural_model,
        "default_env_mode": "neural",  # Use NeRD model
        "device": device
    }
    
    neural_env = NeuralEnvironment(**env_cfg)

@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def basic_pendulum_example(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    exec_config: ExecutionConfig = hydra.utils.instantiate(cfg.execution)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)

    initialize_nerd()

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
