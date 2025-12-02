import newton
import warp as wp

class Model:
    """Adapter for wp.sim.Model -> newton.Model"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.Model(*args, **kwargs)

class State:
    """Adapter for wp.sim.State -> newton.State"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.State(*args, **kwargs)

class Control:
    """Adapter for wp.sim.Control -> newton.Control"""
    def __init__(self, *args, **kwargs):
        self.inner = newton.Control(*args, **kwargs)

class ModelBuilder:
    """Adapter for wp.sim.ModelBuilder -> newton.ModelBuilder"""
    def __init__(self, *args, **kwargs):
        up_vector = kwargs.pop("up_vector", None)
        if up_vector is not None:
            axis = self._convert_up_vector(up_vector)
            kwargs["up_axis"] = axis

        self.inner = newton.ModelBuilder(*args, **kwargs)

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

class JOINT_REVOLUTE:
    """Adapter for wp.sim.JOINT_REVOLUTE -> newton.JointType.REVOLUTE"""
    def __init__(self):
        self.inner = newton.JointType.REVOLUTE

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