import newton
import warp as wp
import types
import numpy as np

import types
import newton

# --- Warp-style GEO constants for legacy code ---
# Warp-style constants
GEO_SPHERE = 0
GEO_CAPSULE = 1
GEO_BOX = 2
GEO_CYLINDER = 3
# Add others if needed


# Map Newton.GeoType -> Warp GEO constants
NEWTON_TO_WARP_GEOTYPE = {
    newton.GeoType.SPHERE: GEO_SPHERE,
    newton.GeoType.CAPSULE: GEO_CAPSULE,
    newton.GeoType.BOX: GEO_BOX,
    newton.GeoType.CYLINDER: GEO_CYLINDER,
}

# Shape properties of geometry
@wp.struct
class ModelShapeGeometry:
    type: wp.array(dtype=wp.int32)  # The type of geometry (GEO_SPHERE, GEO_BOX, etc.)
    is_solid: wp.array(dtype=wp.uint8)  # Indicates whether the shape is solid or hollow
    thickness: wp.array(
        dtype=float
    )  # The thickness of the shape (used for collision detection, and inertia computation of hollow shapes)
    source: wp.array(dtype=wp.uint64)  # Pointer to the source geometry (in case of a mesh, zero otherwise)
    scale: wp.array(dtype=wp.vec3)  # The 3D scale of the shape


# --- GeoTypeAdapter: converts Newton enums to Warp GEO constants ---
class GeoTypeAdapter:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, idx):
        newton_type = self.model.shape_type[idx]
        return NEWTON_TO_WARP_GEOTYPE.get(newton_type, newton_type)

    def __len__(self):
        return self.model.shape_count

class ShapeGeoAdapter:
    """
    Adapter to emulate wp.sim.ModelShapeGeometry for legacy Warp kernels,
    backed by Newton's per-attribute arrays.
    """

    def __init__(self, model):
        self.model = model

        # Map Warp fields to Newton arrays
        self.type = model.shape_type
        self.is_solid = model.shape_is_solid  # bool array in Newton
        self.thickness = getattr(model, "shape_thickness", None)
        self.source = model.shape_source_ptr
        self.scale = model.shape_scale

    def __warp_struct__(self):
        # Return a Warp struct instance for GPU kernels
        return ModelShapeGeometry(
            type=self.type,
            is_solid=self.is_solid,
            thickness=self.thickness,
            source=self.source,
            scale=self.scale,
        )

    def __getitem__(self, idx):
        # Optional: support geo[i] access
        return {
            "type": self.type[idx],
            "is_solid": self.is_solid[idx],
            "thickness": self.thickness[idx] if self.thickness is not None else 0.0,
            "source": self.source[idx],
            "scale": self.scale[idx],
        }

    def __len__(self):
        return len(self.model.shape_type)

# Reuse the original Warp struct definition
@wp.struct
class ModelShapeGeometry:
    type: wp.array(dtype=wp.int32)
    is_solid: wp.array(dtype=wp.uint8)
    thickness: wp.array(dtype=float)
    source: wp.array(dtype=wp.uint64)
    scale: wp.array(dtype=wp.vec3)

class ModelAdapter:
    """
    Adapter for wp.sim.Model -> newton.Model
    Provides:
        - num_envs
        - shape_geo (for legacy kernels)
        - shape_shape_collision
        - geo_types (Warp-style constants)
    """
    def __init__(self, inner_model):
        self.inner = inner_model
        self.num_envs = 1  # emulate Warp environments
        self.shape_shape_collision = np.ones(self.inner.shape_count, dtype=bool)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def shape_geo(self):
        # optional: adapter for shape_geo, see previous examples
        return ShapeGeoAdapter(self.inner).__warp_struct__()
    
    @property
    def geo_types(self):
        return GeoTypeAdapter(self.inner)

###################################################################################################

class Model:
    """Adapter for wp.sim.Model -> newton.Model"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.Model(*args, **kwargs)
        self.num_envs = 1  # start with one "environment"

    def add_builder(self, builder, xform=None, separate_collision_group=True, **kwargs):
        self.inner.add_builder(builder.inner, xform)
        new_shape_count = builder.inner.shape_count if hasattr(builder, "inner") else builder.shape_count
        extra = np.ones(new_shape_count, dtype=bool)
        self.shape_shape_collision = np.concatenate([self.shape_shape_collision, extra])
        # emulate num_envs increment
        self.num_envs += 1


    def __getattr__(self, name):
        # forward all other attributes to inner
        return getattr(self.inner, name)

class State:
    """Adapter for wp.sim.State -> newton.State"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.State(*args, **kwargs)

class Control:
    """Adapter for wp.sim.Control -> newton.Control"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.Control(*args, **kwargs)

###################################################################################################

class ModelBuilder:
    """Adapter for wp.sim.ModelBuilder -> newton.ModelBuilder"""
    def __init__(self, *args, **kwargs):
        up_vector = kwargs.pop("up_vector", None)
        if up_vector is not None:
            axis = self._convert_up_vector(up_vector)
            kwargs["up_axis"] = axis

        self.inner = newton.ModelBuilder(*args, **kwargs)

    def __getattr__(self, name):
        """Forward any unknown attribute to the inner Newton builder."""
        return getattr(self.inner, name)
    
    def add_builder(self, builder, xform=None,
                update_num_env_count=True,
                separate_collision_group=True):

        # (1) ignore update_num_env_count — Newton doesn't have env counts
        # (2) ignore separate_collision_group — Newton has no collision group concept
        # (3) forward call to newton
        return self.inner.add_builder(builder, xform)

    @staticmethod
    def _convert_up_vector(v):
        # assume v is iterable or wp.vec3-like
        x, y, z = float(v[0]), float(v[1]), float(v[2])

        # map vector to closest axis direction
        if abs(x) > abs(y) and abs(x) > abs(z):
            return newton.Axis.X
        elif abs(y) > abs(x) and abs(y) > abs(z):
            return newton.Axis.Y
        else:
            return newton.Axis.Z

    def finalize(self, *args, **kwargs):
        # Call Newton's builder finalize
        model = self.inner.finalize(*args, **kwargs)
        # Wrap the resulting model in a Python adapter for Warp compatibility
        return ModelAdapter(model)  

###################################################################################################

class Mesh:
    """Adapter for wp.sim.Mesh -> newton.Mesh"""
    def __init__(
        self,
        vertices,
        indices,
        normals=None,
        uvs=None,
        compute_inertia=True,
        is_solid=True,
        maxhullvert=32,   # or MESH_MAXHULLVERT constant
        color=None,
        **kwargs
    ):
        # store Newton mesh internally
        self.inner = newton.Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=compute_inertia,
            is_solid=is_solid,
            maxhullvert=maxhullvert,
            color=color,
        )

###################################################################################################


class JOINT_REVOLUTE:
    """Adapter for wp.sim.JOINT_REVOLUTE -> newton.JointType.REVOLUTE"""
    def __init__(self):
        self.inner = newton.JointType.REVOLUTE

###################################################################################################


@wp.kernel
def integrate_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    w: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    gravity: wp.vec3,
    dt: float,
    v_max: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x0 = x[tid]
    v0 = v[tid]

    if (particle_flags[tid] & newton.ParticleFlags.ACTIVE) == 0:
        x_new[tid] = x0
        v_new[tid] = wp.vec3(0.0)
        return

    f0 = f[tid]

    inv_mass = w[tid]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + gravity * wp.step(-inv_mass)) * dt
    # enforce velocity limit to prevent instability
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag
    x1 = x0 + v1 * dt

    x_new[tid] = x1
    v_new[tid] = v1


@wp.func
def integrate_rigid_body(
    q: wp.transform,
    qd: wp.spatial_vector,
    f: wp.spatial_vector,
    com: wp.vec3,
    inertia: wp.mat33,
    inv_mass: float,
    inv_inertia: wp.mat33,
    gravity: wp.vec3,
    angular_damping: float,
    dt: float,
):
    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.quat_rotate(r0, com)

    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping
    w1 *= 1.0 - angular_damping * dt

    q_new = wp.transform(x1 - wp.quat_rotate(r1, com), r1)
    qd_new = wp.spatial_vector(w1, v1)

    return q_new, qd_new


# semi-implicit Euler integration
@wp.kernel
def integrate_bodies(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    I: wp.array(dtype=wp.mat33),
    inv_m: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.mat33),
    gravity: wp.vec3,
    angular_damping: float,
    dt: float,
    # outputs
    body_q_new: wp.array(dtype=wp.transform),
    body_qd_new: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    # masses
    inv_mass = inv_m[tid]  # 1 / mass

    inertia = I[tid]
    inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix

    com = body_com[tid]

    q_new, qd_new = integrate_rigid_body(
        q,
        qd,
        f,
        com,
        inertia,
        inv_mass,
        inv_inertia,
        gravity,
        angular_damping,
        dt,
    )

    body_q_new[tid] = q_new
    body_qd_new[tid] = qd_new

@wp.func
def get_box_vertex(point_id: int, upper: wp.vec3):
    # box vertex numbering:
    #    6---7
    #    |\  |\       y
    #    | 2-+-3      |
    #    4-+-5 |   z \|
    #     \|  \|      o---x
    #      0---1
    # get the vertex of the box given its ID (0-7)
    sign_x = float(point_id % 2) * 2.0 - 1.0
    sign_y = float((point_id // 2) % 2) * 2.0 - 1.0
    sign_z = float((point_id // 4) % 2) * 2.0 - 1.0
    return wp.vec3(sign_x * upper[0], sign_y * upper[1], sign_z * upper[2])

###################################################################################################

class Integrator:
    """
    Generic base class for integrators. Provides methods to integrate rigid bodies and particles.
    """

    def integrate_bodies(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        angular_damping: float = 0.0,
    ):
        """
        Integrate the rigid bodies of the model.

        Args:
            model (Model): The model to integrate.
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).
            angular_damping (float, optional): The angular damping factor. Defaults to 0.0.
        """
        if model.body_count:
            wp.launch(
                kernel=integrate_bodies,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_mass,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.gravity,
                    angular_damping,
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )

    def integrate_particles(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
    ):
        """
        Integrate the particles of the model.

        Args:
            model (Model): The model to integrate.
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).
        """
        if model.particle_count:
            wp.launch(
                kernel=integrate_particles,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    model.particle_inv_mass,
                    model.particle_flags,
                    model.gravity,
                    dt,
                    model.particle_max_velocity,
                ],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=model.device,
            )

    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        """
        Simulate the model for a given time step using the given control input.

        Args:
            model (Model): The model to simulate.
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).
            control (Control): The control input. Defaults to `None` which means the control values from the :class:`Model` are used.
        """
        raise NotImplementedError()

###################################################################################################

class FeatherstoneIntegrator:
    """Adapter for wp.sim.FeatherstoneIntegrator -> newton.solvers.SolverFeatherstone"""
    
    def __init__(self, *args, **kwargs):
        # --- Remap Warp → Newton keyword changes ---
        if "update_mass_matrix_every" in kwargs:
            kwargs["update_mass_matrix_interval"] = kwargs.pop("update_mass_matrix_every")

        self.inner = newton.solvers.SolverFeatherstone(*args, **kwargs)

###################################################################################################

class SimRendererOpenGL:
    """Adapter for wp.sim.render.SimRendererOpenGL -> newton.viewer.ViewerGL"""

    def __init__(self, model, sim_name,
                 up_axis="Z",
                 show_rigid_contact_points=False,
                 contact_points_radius=0.01,
                 show_joints=False,
                 **opengl_render_settings):
        
        # Store Warp-style parameters (for reference or future extensions)
        self.model = model
        self.sim_name = sim_name
        self.up_axis = up_axis
        self.show_rigid_contact_points = show_rigid_contact_points
        self.contact_points_radius = contact_points_radius
        self.show_joints = show_joints

        # Extract only the parameters ViewerGL accepts
        width = opengl_render_settings.get("width", 1920)
        height = opengl_render_settings.get("height", 1080)
        vsync = opengl_render_settings.get("vsync", False)
        headless = opengl_render_settings.get("headless", False)

        # Create the low-level Newton viewer
        self.viewer = newton.viewer.ViewerGL(
            width=width,
            height=height,
            vsync=vsync,
            headless=headless
        )

    def render(self, *args, **kwargs):
        # Warp passes model/state; Newton just calls render()
        # You can extend this to visualize joints/contact points manually
        return self.viewer.render(self.model, *args, **kwargs)

    def update(self, *args, **kwargs):
        # Newton uses step() internally
        if hasattr(self.viewer, "update"):
            return self.viewer.update(*args, **kwargs)
        return self.viewer.step(*args, **kwargs)

    def close(self):
        return self.viewer.close()
    
########################################################
#-----------------Fake-submodules-----------------------
########################################################

render = types.SimpleNamespace(
    SimRendererOpenGL=SimRendererOpenGL
)
