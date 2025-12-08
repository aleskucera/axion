import warp as wp

# =================================================================================
# 1. BAKING (Transposed / SoA)
#    Layout is now: (stride, num_bodies, 6)
#    This ensures Thread(tid) and Thread(tid+1) access adjacent memory.
# =================================================================================


@wp.kernel
def kernel_bake_joints_f16_soa(
    J_flat: wp.array(dtype=wp.spatial_vector),
    body_to_constraints: wp.array(dtype=wp.int32, ndim=2),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    # OUTPUTS (Transposed: Stride x Bodies x 6)
    W_j_f16: wp.array(dtype=wp.float16, ndim=3),
    x_idx_soa: wp.array(dtype=wp.int32, ndim=2),
    stride: int,
):
    body_idx = wp.tid()
    m_inv_local = M_inv[body_idx]

    for i in range(stride):
        flat_idx = body_to_constraints[body_idx, i]

        # SoA Indexing: [i, body_idx] instead of [body_idx * stride + i]
        if flat_idx > 0:
            W_vec = m_inv_local * J_flat[flat_idx]

            # Write components
            W_j_f16[i, body_idx, 0] = wp.float16(W_vec[0])
            W_j_f16[i, body_idx, 1] = wp.float16(W_vec[1])
            W_j_f16[i, body_idx, 2] = wp.float16(W_vec[2])
            W_j_f16[i, body_idx, 3] = wp.float16(W_vec[3])
            W_j_f16[i, body_idx, 4] = wp.float16(W_vec[4])
            W_j_f16[i, body_idx, 5] = wp.float16(W_vec[5])

            x_idx_soa[i, body_idx] = wp.rshift(flat_idx - 1, 1)
        else:
            for k in range(6):
                W_j_f16[i, body_idx, k] = wp.float16(0.0)
            x_idx_soa[i, body_idx] = -1


@wp.kernel
def kernel_bake_contacts_f16_soa(
    J_c: wp.array(dtype=wp.spatial_vector),
    M_inv: wp.array(dtype=wp.spatial_matrix),
    # OUTPUT (Transposed: Stride x Bodies x 6)
    W_c_f16: wp.array(dtype=wp.float16, ndim=3),
    stride: int,
):
    body_idx = wp.tid()
    m_inv_local = M_inv[body_idx]

    # J_c is linear [Body0_C0...C7, Body1_C0...], so we need to calculate read index
    base_read = body_idx * stride

    for i in range(stride):
        idx = base_read + i
        W_vec = m_inv_local * J_c[idx]

        W_c_f16[i, body_idx, 0] = wp.float16(W_vec[0])
        W_c_f16[i, body_idx, 1] = wp.float16(W_vec[1])
        W_c_f16[i, body_idx, 2] = wp.float16(W_vec[2])
        W_c_f16[i, body_idx, 3] = wp.float16(W_vec[3])
        W_c_f16[i, body_idx, 4] = wp.float16(W_vec[4])
        W_c_f16[i, body_idx, 5] = wp.float16(W_vec[5])


# =================================================================================
# 2. GATHER (SoA Reads)
# =================================================================================


def create_gather_f16_soa_kernel(joints_per_body, contacts_per_body):

    @wp.kernel
    def kernel_gather_f16_soa(
        x_joints: wp.array(dtype=wp.float16),
        W_j_f16: wp.array(dtype=wp.float16, ndim=3),  # [Stride, Body, 6]
        x_j_indices: wp.array(dtype=wp.int32, ndim=2),  # [Stride, Body]
        x_contacts: wp.array(dtype=wp.float16),
        W_c_f16: wp.array(dtype=wp.float16, ndim=3),
        v_body: wp.array(dtype=wp.spatial_vector),
    ):
        body_idx = wp.tid()

        # Accumulators (F32)
        v0 = float(0.0)
        v1 = float(0.0)
        v2 = float(0.0)
        v3 = float(0.0)
        v4 = float(0.0)
        v5 = float(0.0)

        # 1. Joints
        for i in range(wp.static(joints_per_body)):
            st_i = wp.static(i)
            # Coalesced Read: indices[i, body_idx]
            # Consecutive threads read consecutive memory addresses here.
            idx = x_j_indices[st_i, body_idx]

            if idx >= 0:
                val = float(x_joints[idx])  # Random access (unavoidable for graph)

                # Coalesced Read: W[i, body_idx, k]
                v0 += float(W_j_f16[st_i, body_idx, 0]) * val
                v1 += float(W_j_f16[st_i, body_idx, 1]) * val
                v2 += float(W_j_f16[st_i, body_idx, 2]) * val
                v3 += float(W_j_f16[st_i, body_idx, 3]) * val
                v4 += float(W_j_f16[st_i, body_idx, 4]) * val
                v5 += float(W_j_f16[st_i, body_idx, 5]) * val

        # 2. Contacts
        # Need to know the linear index for x_contacts
        base_c = body_idx * contacts_per_body

        for i in range(wp.static(contacts_per_body)):
            st_i = wp.static(i)
            # x_contacts is dense/linear.
            # Thread 0 reads x[0], Thread 1 reads x[32]...
            # This 'x' read is still strided, but W is now coalesced.
            # Since x is 2 bytes, it's less painful than W (12 bytes).
            val = float(x_contacts[base_c + st_i])

            v0 += float(W_c_f16[st_i, body_idx, 0]) * val
            v1 += float(W_c_f16[st_i, body_idx, 1]) * val
            v2 += float(W_c_f16[st_i, body_idx, 2]) * val
            v3 += float(W_c_f16[st_i, body_idx, 3]) * val
            v4 += float(W_c_f16[st_i, body_idx, 4]) * val
            v5 += float(W_c_f16[st_i, body_idx, 5]) * val

        v_body[body_idx] = wp.spatial_vector(v0, v1, v2, v3, v4, v5)

    return kernel_gather_f16_soa


# =================================================================================
# 3. APPLY (Unchanged - Linear)
# =================================================================================


@wp.kernel
def kernel_apply_joints_f16(
    v_body: wp.array(dtype=wp.spatial_vector),
    J_flat: wp.array(dtype=wp.spatial_vector),
    constraint_to_body: wp.array(dtype=wp.int32, ndim=2),
    C: wp.array(dtype=wp.float16),
    x: wp.array(dtype=wp.float16),
    z: wp.array(dtype=wp.float16),
):
    c_idx = wp.tid()
    flat_idx_0 = 1 + c_idx * 2
    flat_idx_1 = 1 + c_idx * 2 + 1

    b0 = constraint_to_body[c_idx, 0]
    b1 = constraint_to_body[c_idx, 1]

    res = float(0.0)
    if b0 >= 0:
        res += wp.dot(J_flat[flat_idx_0], v_body[b0])
    if b1 >= 0:
        res += wp.dot(J_flat[flat_idx_1], v_body[b1])

    res += float(C[c_idx]) * float(x[c_idx])
    z[c_idx] = wp.float16(res)


def create_apply_contacts_f16_kernel(stride):
    @wp.kernel
    def kernel_apply_contacts_f16(
        v_body: wp.array(dtype=wp.spatial_vector),
        J_c: wp.array(dtype=wp.spatial_vector),
        C: wp.array(dtype=wp.float16),
        x: wp.array(dtype=wp.float16),
        z: wp.array(dtype=wp.float16),
    ):
        c_idx = wp.tid()
        b_idx = c_idx // stride
        val = wp.dot(J_c[c_idx], v_body[b_idx])
        val += float(C[c_idx]) * float(x[c_idx])
        z[c_idx] = wp.float16(val)

    return kernel_apply_contacts_f16
