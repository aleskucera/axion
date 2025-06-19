import warp as wp


@wp.kernel
def _add_inplace_f_f_kernel(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    offset_a: wp.int32,
    offset_b: wp.int32,
):
    tid = wp.tid()
    wp.atomic_add(a, offset_a + tid, b[offset_b + tid])


@wp.kernel
def _add_inplace_sv_sv_kernel(
    a: wp.array(dtype=wp.spatial_vector),
    b: wp.array(dtype=wp.spatial_vector),
    offset_a: wp.int32,
    offset_b: wp.int32,
):
    tid = wp.tid()
    wp.atomic_add(a, offset_a + tid, b[offset_b + tid])


@wp.kernel
def _add_inplace_sv_f_kernel(
    a: wp.array(dtype=wp.spatial_vector),
    b: wp.array(dtype=wp.float32),
    offset_a: wp.int32,
    offset_b: wp.int32,
):
    tid = wp.tid()
    vec_idx = tid // 6
    component_idx = tid % 6
    value_to_add = b[offset_b + tid]
    delta_vec = wp.spatial_vector()
    delta_vec[component_idx] = value_to_add
    wp.atomic_add(a, offset_a + vec_idx, delta_vec)


def add_inplace(a: wp.array, b: wp.array, offset_a: int, offset_b: int, count: int):
    if count <= 0:
        return
    a_is_sv = a.dtype == wp.spatial_vector
    b_is_sv = b.dtype == wp.spatial_vector
    kernel = None
    dim = count
    inputs = [a, b, offset_a, offset_b]
    if a_is_sv:
        if b_is_sv:
            kernel = _add_inplace_sv_sv_kernel
        else:  # spatial_vector += float
            kernel = _add_inplace_sv_f_kernel
            dim = count * 6
    else:  # float += ...
        if b_is_sv:
            raise NotImplementedError(
                "Adding a spatial_vector array to a float array is not implemented."
            )
        else:  # float += float
            kernel = _add_inplace_f_f_kernel
    if kernel is None:
        raise TypeError(
            f"Unsupported dtype combination for add_inplace: a={a.dtype}, b={b.dtype}"
        )
    wp.launch(kernel=kernel, dim=dim, inputs=inputs, device=a.device)
