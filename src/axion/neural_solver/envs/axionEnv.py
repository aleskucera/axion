"""
Axion-based environment implementing the NeRD/Warp env contract.

This class is intended to back `AxionEnvToTrajectorySamplerAdapter` without
changing its API. It exposes the same attributes and methods that the
original NeRD env provided, but runs dynamics through the Axion/Newton
engine stack.
"""

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

# Approximate penetration depth from endpoints and normal.
# depth = dot(point1 - point0, normal)
# @wp.kernel
# def compute_depth(
#     p0: wp.array(dtype=wp.vec3, ndim=2),
#     p1: wp.array(dtype=wp.vec3, ndim=2),
#     n: wp.array(dtype=wp.vec3, ndim=2),
#     depth: wp.array(dtype=float, ndim=2),
# ):
#     w = wp.tid() // p0.shape[1]
#     i = wp.tid() % p0.shape[1]
#     d = wp.dot(p1[w, i] - p0[w, i], n[w, i])
#     depth[w, i] = d


@wp.kernel
def compute_depth_flat(
    p0: wp.array(dtype=wp.vec3),
    p1: wp.array(dtype=wp.vec3),
    n: wp.array(dtype=wp.vec3),
    depth: wp.array(dtype=float),
):
    i = wp.tid()
    depth[i] = wp.dot(p1[i] - p0[i], n[i])


# Copy batched (num_worlds, num_contacts_per_env) -> flat (num_total_contacts) for NeRD-style 1D API.
@wp.kernel
def batched_to_flat_kernel(
    num_contacts_per_env: wp.int32,
    batched_point0: wp.array(dtype=wp.vec3, ndim=2),
    batched_point1: wp.array(dtype=wp.vec3, ndim=2),
    batched_normal: wp.array(dtype=wp.vec3, ndim=2),
    batched_shape0: wp.array(dtype=wp.int32, ndim=2),
    batched_shape1: wp.array(dtype=wp.int32, ndim=2),
    batched_thickness: wp.array(dtype=wp.float32, ndim=2),
    flat_point0: wp.array(dtype=wp.vec3),
    flat_point1: wp.array(dtype=wp.vec3),
    flat_normal: wp.array(dtype=wp.vec3),
    flat_shape0: wp.array(dtype=wp.int32),
    flat_shape1: wp.array(dtype=wp.int32),
    flat_thickness: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    world = tid // num_contacts_per_env
    slot = tid % num_contacts_per_env
    flat_point0[tid] = batched_point0[world, slot]
    flat_point1[tid] = batched_point1[world, slot]
    flat_normal[tid] = batched_normal[world, slot]
    flat_shape0[tid] = batched_shape0[world, slot]
    flat_shape1[tid] = batched_shape1[world, slot]
    flat_thickness[tid] = batched_thickness[world, slot]


# Pendulum geometry (match examples/pendulum_AxionEngine.py)
PENDULUM_HEIGHT = 5.0

def build_pendulum_model(
    num_worlds: int,
    device: wp.Device,
    requires_grad: bool = False,
) -> newton.Model:
    """Build the same 2-link revolute pendulum as examples/pendulum_AxionEngine.py, replicated for num_worlds."""
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


def build_model_from_env_name(
    env_name: str,
    num_envs: int,
    device: wp.Device,
    requires_grad: bool,
) -> newton.Model:
    """Dispatch to the appropriate model builder for env_name."""
    if env_name in ("PendulumWithContact", "Pendulum", "pendulum"):
        return build_pendulum_model(
            num_worlds=num_envs,
            device=device,
            requires_grad=requires_grad,
        )
    raise ValueError(
        f"Unknown env_name={env_name!r}. "
        "Supported: 'PendulumWithContact', 'Pendulum', 'pendulum'."
    )

@dataclass
class _AxionEnvConfig:
    env_name: str
    num_envs: int
    dt: float
    device: wp.Device
    requires_grad: bool = False


class AxionEnv:
    """
    Axion environment providing the minimal API required by:
      - `AxionEnvToTrajectorySamplerAdapter`
      - `TrajectorySampler` / `TrajectorySamplerPendulum`
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        requires_grad: bool = False,
        device: str = "cuda:0",
        render: bool = False,
        **kwargs: Any,
    ) -> None:
        del render  # Rendering is handled at a higher level if needed.

        self._cfg = _AxionEnvConfig(
            env_name=env_name,
            num_envs=num_envs,
            dt=1.0e-3,
            device=wp.get_device(device),
            requires_grad=requires_grad,
        )

        # Build Newton model from env_name (multi-world).
        self.model = build_model_from_env_name(
            env_name=env_name,
            num_envs=num_envs,
            device=self._cfg.device,
            requires_grad=requires_grad,
        )

        # Core simulation objects
        self.state: newton.State = self.model.state()
        self._next_state: newton.State = self.model.state()
        self.control: newton.Control = self.model.control()

        # Contacts and Axion engine
        rigid_max = max(
            self.model.rigid_contact_max,
            self.model.num_worlds * 32,
        )
        self._contacts = newton.Contacts(
            rigid_max,
            0,
            requires_grad=requires_grad,
            device=self._cfg.device,
        )
        self._contacts_abstract = newton.Contacts(
            rigid_max,
            0,
            requires_grad=requires_grad,
            device=self._cfg.device,
        )
        self._axion_contacts = AxionContacts(
            model=self.model,
            max_contacts_per_world=32,
        )

        self._engine_cfg = AxionEngineConfig()
        self._engine = AxionEngine(
            model=self.model,
            init_state_fn=self._init_state_fn,
            config=self._engine_cfg,
        )

        # Contract attributes
        self.device = self._cfg.device
        self.frame_dt = self._cfg.dt
        self.eval_collisions: bool = True
        self.robot_name: str = env_name

        # Joint/state dimensions per env
        self.dof_q_per_env = int(self.model.joint_coord_count // self.model.num_worlds)
        self.dof_qd_per_env = int(self.model.joint_dof_count // self.model.num_worlds)

        self.bodies_per_env = int(self.model.body_count // self.model.num_worlds)

        # Control dimensions (simple force-mode control)
        self.joint_act_dim = int(self.model.joint_dof_count // self.model.num_worlds)
        self.control_dim = self.joint_act_dim

        # Indexable limits for WarpSimDataGenerator (action_limits[i][0], action_limits[i][1])
        self._control_limits_np = np.full((self.control_dim, 2), [-1.0, 1.0], dtype=np.float32)
        self.control_limits = self._control_limits_np
        self.control_limits_wp = wp.array(
            self._control_limits_np,
            dtype=wp.float32,
            device=self.device,
        )

        # Simple identity mapping: all DOFs are controllable with unit gain.
        self.controllable_dofs_wp = wp.array(
            np.arange(self.joint_act_dim, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.control_gains_wp = wp.array(
            np.ones(self.control_dim, dtype=np.float64),
            dtype=wp.float64,
            device=self.device,
        )

        # Temporary joint_act buffer exposed to the adapter.
        self.joint_act = wp.zeros(
            self.model.joint_dof_count,
            dtype=float,
            device=self.device,
        )
        # Mirror inside control for adapter compatibility.
        self.control.joint_act = self.joint_act

        # Abstract contacts view
        self.abstract_contacts = _AxionAbstractContacts(
            model=self.model,
            axion_contacts=self._axion_contacts,
        )

        # Track whether generalized coordinates are used (Newton does).
        self.uses_generalized_coordinates = True

    # --------------------------------------------------------------------- #
    # Methods required by the adapter and samplers
    # --------------------------------------------------------------------- #

    @property
    def num_envs(self) -> int:
        return self._cfg.num_envs

    def set_eval_collisions(self, eval_collisions: bool) -> None:
        self.eval_collisions = eval_collisions

    def _init_state_fn(
        self,
        current_state: newton.State,
        next_state: newton.State,
        contacts: newton.Contacts,
        dt: float,
    ) -> None:
        # Simple integrator hook: rely on AxionEngine to integrate bodies.
        self._engine.integrate_bodies(self.model, current_state, next_state, dt)

    def assign_control(
        self,
        actions_wp: wp.array,
        control: newton.Control,
        state: newton.State,
    ) -> None:
        # Directly treat actions as generalized forces on each joint DOF.
        # Copy to joint_act buffer then into control.joint_f.
        wp.copy(dest=self.joint_act, src=actions_wp)
        if control.joint_f is not None:
            wp.copy(dest=control.joint_f, src=self.joint_act)

    def update(self) -> None:
        # Clear accumulated forces.
        self.state.clear_forces()

        # Acquire contacts either from geometry or from abstract override.
        if self.eval_collisions:
            self._contacts = self.model.collide(self.state)
            self._axion_contacts.load_contact_data(self._contacts)
            self.abstract_contacts.update_from_axion()
        else:
            # In abstract mode: copy flat abstract_contacts into Newton Contacts
            # so the engine uses them in _initialize_constraints(contacts).
            ac = self.abstract_contacts
            n = ac.num_total_contacts
            self._contacts_abstract.clear()
            wp.copy(self._contacts_abstract.rigid_contact_point0, ac.contact_point0)
            wp.copy(self._contacts_abstract.rigid_contact_point1, ac.contact_point1)
            wp.copy(self._contacts_abstract.rigid_contact_normal, ac.contact_normal)
            wp.copy(self._contacts_abstract.rigid_contact_shape0, ac.contact_shape0)
            wp.copy(self._contacts_abstract.rigid_contact_shape1, ac.contact_shape1)
            wp.copy(self._contacts_abstract.rigid_contact_thickness0, ac.contact_thickness)
            self._contacts_abstract.rigid_contact_thickness1.zero_()
            wp.copy(
                self._contacts_abstract.rigid_contact_count,
                wp.array([n], dtype=wp.int32, device=self.device),
            )
            self._contacts = self._contacts_abstract

        # Single Axion engine step.
        self._engine.step(
            state_in=self.state,
            state_out=self._next_state,
            control=self.control,
            contacts=self._contacts,
            dt=self.frame_dt,
        )

        self.state, self._next_state = self._next_state, self.state

    def reset(self) -> None:
        # Zero joint coordinates and velocities; run FK to update body_q.
        self.state.joint_q.zero_()
        self.state.joint_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def render(self) -> None:
        # Rendering is not handled here; adapter's render() is a no-op.
        return

    def close(self) -> None:
        # Nothing to clean up explicitly.
        return


class _AxionAbstractContacts:
    """
    Adapter exposing AxionContacts in the NeRD abstract_contacts format:
    flat 1D arrays (num_total_contacts,) so kernels like compute_contact_points_0_world
    can use contact_id = wp.tid() and index contact_shape0[contact_id], etc.
    """

    def __init__(self, model: newton.Model, axion_contacts: AxionContacts) -> None:
        self._model = model
        self._axion_contacts = axion_contacts
        self.device = model.device

        self.num_worlds = model.num_worlds
        self.num_contacts_per_env = axion_contacts.max_contacts
        self.num_total_contacts = self.num_worlds * self.num_contacts_per_env

        with wp.ScopedDevice(self.device):
            self.contact_point0 = wp.zeros(
                self.num_total_contacts,
                dtype=wp.vec3,
            )
            self.contact_point1 = wp.zeros(
                self.num_total_contacts,
                dtype=wp.vec3,
            )
            self.contact_normal = wp.zeros(
                self.num_total_contacts,
                dtype=wp.vec3,
            )
            self.contact_shape0 = wp.zeros(
                self.num_total_contacts,
                dtype=wp.int32,
            )
            self.contact_shape1 = wp.zeros(
                self.num_total_contacts,
                dtype=wp.int32,
            )
            self.contact_thickness = wp.zeros(
                self.num_total_contacts,
                dtype=wp.float32,
            )
            self.contact_depth = wp.zeros(
                self.num_total_contacts,
                dtype=wp.float32,
            )

    def update_from_axion(self) -> None:
        # Copy from AxionContacts (batched 2D) into flat 1D abstract view.
        total = self.num_total_contacts
        wp.launch(
            batched_to_flat_kernel,
            dim=total,
            inputs=[
                self.num_contacts_per_env,
                self._axion_contacts.contact_point0,
                self._axion_contacts.contact_point1,
                self._axion_contacts.contact_normal,
                self._axion_contacts.contact_shape0,
                self._axion_contacts.contact_shape1,
                self._axion_contacts.contact_thickness0,
            ],
            outputs=[
                self.contact_point0,
                self.contact_point1,
                self.contact_normal,
                self.contact_shape0,
                self.contact_shape1,
                self.contact_thickness,
            ],
            device=self.device,
        )
        wp.launch(
            compute_depth_flat,
            dim=total,
            inputs=[self.contact_point0, self.contact_point1, self.contact_normal],
            outputs=[self.contact_depth],
            device=self.device,
        )
