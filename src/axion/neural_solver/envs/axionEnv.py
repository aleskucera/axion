from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import warp as wp
import newton

from axion.core.contacts import AxionContacts
from axion import JointMode
from axion.core.engine import AxionEngine
from axion.core.engine_config import AxionEngineConfig
from axion.core.model_builder import AxionModelBuilder

from axion.neural_solver.utils.warp_utils import device_to_torch
from axion.neural_solver.envs.abstract_contact import AbstractContact

PENDULUM_HEIGHT = 5.0
NUM_CONTACTS_PER_ENV = 4 # hardcoded for double pendulum
FRAME_DT = 0.01
ENGINE_SUBSTEPS = 10
ENGINE_DT = FRAME_DT/ENGINE_SUBSTEPS

class AxionEnv:
    """
    Wrapper around Axion. Currently it builds the double pendulum model and calls Axion as an integrator.
    Instance of AxionEnv will be created inside axionToTrajectorySampler class.  
    """
    
    def __init__(
        self,
        env_name: str,
        num_worlds: int,
        device: wp.Device,
        requires_grad: bool,
        ):
        
        self.env_name = env_name
        self.num_worlds = num_worlds
        self.device = wp.get_device(device)
        self.requires_grad = requires_grad
        self.robot_name: str = self.env_name    # for compatibility with AxionToTrajectorySampler 
        
        # model (robot model built by AxionModelBuilder):
        if self.env_name in ("PendulumWithContact", "Pendulum", "pendulum"):
            self.model = self._build_pendulum_model(self.num_worlds, self.device, self.requires_grad)
        else:
            raise NotImplementedError
        
        # value "inherited" from the robot model:
        self.state: newton.State = self.model.state()   #current state
        self.next_state: newton.State = self.model.state()
        self.control: newton.Control = self.model.control()
        self.contacts: newton.Contacts = self.model.collide(self.state)

        # integrator (Axion engine):
        # Use the same config as examples/conf/engine/axion_pos.yaml
        # so that dataset generation matches the interactive simulator.
        self.engine_cfg = AxionEngineConfig(
            max_newton_iters=12,
            max_linear_iters=16,
            enable_linesearch=True,
            linesearch_conservative_step_count=16,
            linesearch_conservative_upper_bound=5e-2,
            linesearch_min_step=1e-6,
            linesearch_optimistic_step_count=48,
            linesearch_optimistic_window=0.4,
            joint_stabilization_factor=0.01,
            contact_stabilization_factor=0.02,
            joint_compliance=6e-8,
            equality_compliance=1e-7,
            contact_compliance=1e-6,
            friction_compliance=1e-6,
            regularization=1e-6,
            contact_fb_alpha=0.5,
            contact_fb_beta=1.0,
            friction_fb_alpha=1.0,
            friction_fb_beta=1.0,
            max_contacts_per_world=256,
            joint_constraint_level="pos",
            contact_constraint_level="pos",
        )
        self.engine = AxionEngine(
            model=self.model,
            init_state_fn=self._engine_init_state_fn,
            config=self.engine_cfg,
        )

        # number of DOF info
        self.dof_q_per_env = int(self.model.joint_coord_count / self.model.num_worlds)  # 2 for planar double pendulum
        self.dof_qd_per_env = int(self.model.joint_dof_count / self.model.num_worlds)   # 2 for planar double pendulum
        self.bodies_per_env = int(self.model.body_count / self.model.num_worlds)
        
        # control info
        self.joint_act_dim = self.dof_q_per_env # 2 for planar double pendulum
        self.control_dim = self.joint_act_dim
        self.control_limits = np.full((self.control_dim, 2), [-1.0, 1.0], dtype=np.float32)
        # Temporary joint_act buffer exposed to the adapter.
        self.joint_act = wp.zeros(
            self.model.joint_dof_count,
            dtype=float,
            device=self.device,
        )
        # Mirror inside control for adapter compatibility - Newton's Control does not have joint_act
        self.control.joint_act = self.joint_act     # implicitly creates joint_act!! not  ideal
        
        # contact info
        self.abstract_contacts = AbstractContact(
            num_contacts_per_env = NUM_CONTACTS_PER_ENV ,
            num_envs = self.num_envs,
            model = self.model, 
            device = device_to_torch(self.model.device)
        )
        self.eval_collisions: bool = True   # ?

        # misc
        self.frame_dt = FRAME_DT
        self.uses_generalized_coordinates = False  # AxionEngine is a maximal-coordinate solver


    def set_eval_collisions(self, eval_collisions: bool) -> None:
        self.eval_collisions = eval_collisions

    def update(self) -> None:
        """
        Triggers (Axion's) collision detection and step Axion Engine
        """

        self.state.clear_forces()

        # for i in range(ENGINE_SUBSTEPS):
        #     # Collision detection
        #     if self.eval_collisions:
        #         self.contacts = self.model.collide(self.state)
        #     else:
        #         raise NotImplementedError
            
        #     # Engine (integrator) step
        #     self.engine.step(
        #         state_in = self.state,
        #         state_out = self.next_state,
        #         control= self.control,
        #         contacts=self.contacts, 
        #         dt= ENGINE_DT
        #     )

        if self.eval_collisions:
            self.contacts = self.model.collide(self.state)
        else:
            raise NotImplementedError       

        # Engine (integrator) step
        self.engine.step(
            state_in = self.state,
            state_out = self.next_state,
            control= self.control,
            contacts=self.contacts, 
            dt= FRAME_DT
        )

        self.state, self.next_state = self.next_state, self.state

        # AxionEngine is a maximal-coordinate solver: it only writes body_q / body_qd.
        # Recover generalized coordinates so that joint_q / joint_qd are up-to-date.
        newton.eval_ik(self.model, self.state, self.state.joint_q, self.state.joint_qd)

    def assign_control(
        self,
        actions_wp: wp.array,
        control: newton.Control,
        state: newton.State,
    ) -> None:
        # The purpose of this function was to write the "actions_wp" from input arguments
        # into control.joint_act (the self.control.joint_act was passed to that).
        # Anyway, this is a do-nothing function for now, because Newton's model does not
        # have joint_act and we expose this only for compatibility purposes
        return

    def reset(self) -> None:
        # Zero joint coordinates and velocities; run FK to update body_q.
        self.state.joint_q.zero_()
        self.state.joint_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def close(self) -> None:
        # Nothing to clean up explicitly.
        return

    # "Private" methods:

    def _build_pendulum_model(
        self,
        num_worlds: int,
        device: wp.Device,
        requires_grad: bool = False,
    ) -> newton.Model:
        """Build the same 2-link revolute pendulum as examples/pendulum_AxionEngine.py,
        replicated for num_worlds."""
        builder = AxionModelBuilder()

        chain_width = 1.5
        shape_ke = 1.0e4
        shape_kd = 1.0e3
        shape_kf = 1.0e4
        hx = chain_width * 0.5

        link_config = newton.ModelBuilder.ShapeConfig(
            density=500.0, ke=shape_ke, kd=shape_kd, kf=shape_kf
        )
        capsule_xform = wp.transform(
            p=wp.vec3(0.0, 0.0, 0.0),
            q=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi / 2),
        )

        link_0 = builder.add_link(armature=0.1)
        builder.add_shape_capsule(
            link_0,
            xform=capsule_xform,
            radius=0.1,
            half_height=chain_width * 0.5,
            cfg=link_config,
        )

        link_1 = builder.add_link(armature=0.1)
        builder.add_shape_capsule(
            link_1,
            xform=capsule_xform,
            radius=0.1,
            half_height=chain_width * 0.5,
            cfg=link_config,
        )

        j0 = builder.add_joint_revolute(
            parent=-1,
            child=link_0,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, PENDULUM_HEIGHT), q=wp.quat_identity()
            ),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=1000.0,
            target_kd=50.0,
            custom_attributes={
                #"joint_target_ki": [0.5],
                "joint_dof_mode": [JointMode.NONE],
            },
        )
        j1 = builder.add_joint_revolute(
            parent=link_0,
            child=link_1,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            target_ke=500.0,
            target_kd=5.0,
            custom_attributes={
                #"joint_target_ki": [0.5],
                "joint_dof_mode": [JointMode.NONE],
            },
            armature=0.1,
        )

        builder.add_articulation([j0, j1], key="pendulum")
        builder.add_ground_plane()

        return builder.finalize_replicated(
            num_worlds=num_worlds,
            gravity=-9.81,
            device=device,
            requires_grad=requires_grad,
        )

    def _engine_init_state_fn(
        self,
        state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ) -> None:
        # Simple integrator hook: rely on AxionEngine to integrate bodies.
        self.engine.integrate_bodies(self.model, state, next_state, dt)

    # Properties:
    @property
    def num_envs(self) -> int:
        return self.num_worlds
    
