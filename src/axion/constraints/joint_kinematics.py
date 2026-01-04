import warp as wp
from axion.math import orthogonal_basis


@wp.func
def compute_joint_transforms(
    body_q: wp.transform,
    body_com: wp.vec3,
    joint_X_local: wp.transform,
):
    """
    Computes the World Space joint frame and the lever arm (vector from COM to Joint).
    """
    # Joint Frame in World Space: X_w = X_body * X_local
    X_w = body_q * joint_X_local
    
    # Center of Mass in World Space
    com_w = wp.transform_point(body_q, body_com)
    
    # Joint Position in World Space
    pos_w = wp.transform_get_translation(X_w)

    # Lever Arm: r = pos_joint - pos_com
    r = pos_w - com_w

    return X_w, r, pos_w


# ---------------------------------------------------------------------------- #
#                               Constraint Helpers                             #
# ---------------------------------------------------------------------------- #

@wp.func
def get_linear_component(
    r_p: wp.vec3,
    r_c: wp.vec3,
    pos_p: wp.vec3,
    pos_c: wp.vec3,
    axis_idx: wp.int32,  # 0=X, 1=Y, 2=Z
):
    """
    Generates the Jacobian data and Error for a linear constraint along a global axis.
    Used by: Spherical, Revolute, Fixed.
    """
    # 1. Define the Global Axis (World Space)
    axis_vec = wp.vec3(0.0, 0.0, 0.0)
    if axis_idx == 0:
        axis_vec = wp.vec3(1.0, 0.0, 0.0)
    elif axis_idx == 1:
        axis_vec = wp.vec3(0.0, 1.0, 0.0)
    else:
        axis_vec = wp.vec3(0.0, 0.0, 1.0)

    # 2. Compute Jacobian (Linear velocity part)
    # J_linear = axis_vec
    # J_angular = r x axis_vec
    #
    # We return the Spatial Jacobian components:
    # J_lin is the force direction (axis_vec)
    # J_ang is the torque direction (r x axis_vec)
    
    ang_p = wp.cross(r_p, axis_vec)
    ang_c = wp.cross(r_c, axis_vec)

    # For the parent, the force is usually -axis_vec, but we standardizing on
    # returning the "Positive" direction. The solver handles the sign 
    # (Child - Parent) or similar.
    # Here we return the specific spatial vectors for Child and Parent 
    # assuming the constraint is: C(x) = x_c - x_p = 0
    
    # Parent Term: -1 * (v_p + w_p x r_p)
    # J_p_lin = -axis_vec
    # J_p_ang = -cross(r_p, axis_vec)
    
    # Child Term: +1 * (v_c + w_c x r_c)
    # J_c_lin = axis_vec
    # J_c_ang = cross(r_c, axis_vec)

    J_c = wp.spatial_vector(axis_vec, ang_c)
    J_p = wp.spatial_vector(-axis_vec, -ang_p)

    # 3. Compute Error (Distance)
    delta = pos_c - pos_p
    error = delta[axis_idx]

    return J_p, J_c, error


@wp.func
def get_angular_component(
    X_wp: wp.transform,
    X_wc: wp.transform,
    axis_idx: wp.int32, # 0, 1, or 2 relative to the Joint Frame
):
    """
    Generates the Jacobian data and Error for an angular constraint.
    This locks the rotation around a specific local axis of the joint.
    
    Used by: Fixed (locks 0, 1, 2), Revolute (locks 0, 1 OR 1, 2 depending on definition).
    """
    # Extract Rotations
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # We want to lock relative rotation.
    # Conceptually: The Child frame should align with the Parent frame 
    # (possibly with some initial offset, but X_wp and X_wc include that).
    
    # Let's define the constraint based on aligning axes.
    # To lock 3DOF rotation, we need X_p=X_c, Y_p=Y_c, Z_p=Z_c.
    
    # General strategy for small angles (velocity level):
    # w_c - w_p = 0
    # J_c = [0, axis], J_p = [0, -axis]
    
    # Which axis? The constraint is applied along the World Space axes 
    # corresponding to the current orientation of the joint.
    # Effectively, we are constraining the relative angular velocity along 
    # the local axis_idx transformed to world space.
    
    # Get the axis in World Space (using Parent frame as reference is standard)
    local_axis = wp.vec3(0.0, 0.0, 0.0)
    if axis_idx == 0:
        local_axis = wp.vec3(1.0, 0.0, 0.0)
    elif axis_idx == 1:
        local_axis = wp.vec3(0.0, 1.0, 0.0)
    else:
        local_axis = wp.vec3(0.0, 0.0, 1.0)
        
    axis_w = wp.quat_rotate(q_p, local_axis)

    # Jacobian (Pure Angular)
    # J_c = [0, axis_w]
    # J_p = [0, -axis_w]
    J_c = wp.spatial_vector(wp.vec3(0.0), axis_w)
    J_p = wp.spatial_vector(wp.vec3(0.0), -axis_w)

    # Error (Orientation difference)
    # We can approximate this as the component of the relative rotation vector.
    # q_rel = q_p^-1 * q_c
    # 2 * V(q_rel) ~= theta * axis
    
    q_p_inv = wp.quat_inverse(q_p)
    q_rel = wp.mul(q_p_inv, q_c)
    
    # The imaginary part of q_rel (x, y, z) corresponds to sin(theta/2) * axis.
    # For small angles, this is approx theta/2 * axis.
    # We multiply by 2 to get theta.
    
    # Note: Warp quats are (x, y, z, w)
    vec_part = wp.vec3(q_rel[0], q_rel[1], q_rel[2])
    
    # Error along the specific axis
    # error = 2 * dot(vec_part, local_axis)
    # Note: We use local_axis here because q_rel is in the local frame of P.
    error = 2.0 * vector_dot_axis(vec_part, axis_idx)
    
    # Stability check: If w is negative, we are taking the "long way" around.
    # Flip to ensure shortest path.
    if q_rel[3] < 0.0:
        error = -error
        
    return J_p, J_c, error


@wp.func
def get_revolute_angular_component(
    X_wp: wp.transform,
    X_wc: wp.transform,
    hinge_axis_local: wp.vec3,
    ortho_idx: wp.int32, # 0 or 1
):
    """
    Specialized helper for Revolute joints.
    Instead of locking X/Y/Z, it locks the two axes ORTHOGONAL to the hinge axis.
    Uses geometric formulation (dot product) to match the Jacobian.
    """
    # 1. Transform Key Vectors to World Space
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # The hinge axis in World Space (Parent)
    axis_p_world = wp.quat_rotate(q_p, hinge_axis_local)

    # The basis vectors perpendicular to the hinge (in Local Space of P)
    b1_local, b2_local = orthogonal_basis(hinge_axis_local)

    # Which orthogonal basis vector are we checking against?
    # We want the Child's Hinge Axis to stay aligned with Parent's Hinge Axis.
    # So we check if Child's Hinge is orthogonal to Parent's Ortho Basis?
    # No, that allows Child to rotate around Ortho Basis.
    
    # Correct Geometric Constraint for Hinge Alignment:
    # Child's Ortho Vectors should rotate with the Hinge? No, child rotates AROUND hinge.
    # The constraint is that the Hinge Axis itself must match.
    # i.e. Child's local Hinge Axis (transformed to World) must == Parent's Hinge Axis (World).
    # This 2-DOF constraint is enforced by saying:
    # Child's Hinge Axis must be orthogonal to Parent's Basis vectors b1 and b2.
    
    # Wait, the previous implementation locked "Parent's Basis vs Child's Basis"?
    # That locks the TWIST as well (making it Fixed).
    # A Hinge Joint allows rotation around the Hinge Axis.
    # So we must NOT constrain the basis vectors that spin around the hinge.
    # We only constrain the AXIS direction.
    
    # Constrain: dot(axis_child_world, b1_parent_world) = 0
    # Constrain: dot(axis_child_world, b2_parent_world) = 0
    
    axis_c_world = wp.quat_rotate(q_c, hinge_axis_local)
    
    # Parent Basis World
    target_basis_p_world = wp.vec3(0.0)
    if ortho_idx == 0:
        target_basis_p_world = wp.quat_rotate(q_p, b1_local)
    else:
        target_basis_p_world = wp.quat_rotate(q_p, b2_local)
        
    # Error: Projection of Child Axis onto Parent's Forbidden Planes
    error = wp.dot(axis_c_world, target_basis_p_world)
    
    # Jacobian:
    # C = a_c . b_p
    # dC/dt = da_c/dt . b_p + a_c . db_p/dt
    # da_c/dt = w_c x a_c
    # db_p/dt = w_p x b_p
    # dC/dt = (w_c x a_c) . b_p + a_c . (w_p x b_p)
    #       = w_c . (a_c x b_p) + w_p . (b_p x a_c)
    #       = (a_c x b_p) . (w_c - w_p)
    
    # Rot Axis for Jacobian = cross(axis_c_world, target_basis_p_world)
    rot_axis = wp.cross(axis_c_world, target_basis_p_world)
    
    J_c = wp.spatial_vector(wp.vec3(0.0), rot_axis)
    J_p = wp.spatial_vector(wp.vec3(0.0), -rot_axis)

    return J_p, J_c, error


@wp.func
def vector_dot_axis(v: wp.vec3, axis_idx: wp.int32):
    if axis_idx == 0:
        return v[0]
    elif axis_idx == 1:
        return v[1]
    return v[2]
