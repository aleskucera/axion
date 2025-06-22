# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from math import sqrt
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import warp as wp
import warp.sparse as sparse
from warp.optim.linear import _Matrix
from warp.optim.linear import aslinearoperator
from warp.optim.linear import LinearOperator
from warp.optim.linear import preconditioner
from warp.utils import array_inner

# No need to auto-generate adjoint code for linear solvers
wp.set_module_options({"enable_backward": False})


def cr(
    A: _Matrix,
    b: wp.array,
    x: wp.array,
    tol: Optional[float] = None,
    atol: Optional[float] = None,
    maxiter: Optional[float] = 0,
    M: Optional[_Matrix] = None,
    callback: Optional[Callable] = None,
    check_every=10,
    use_cuda_graph=True,
) -> Tuple[int, float, float]:
    """Computes an approximate solution to a symmetric, positive-definite linear system
    using the Conjugate Residual algorithm.

    Args:
        A: the linear system's left-hand-side
        b: the linear system's right-hand-side
        x: initial guess and solution vector
        tol: relative tolerance for the residual, as a ratio of the right-hand-side norm
        atol: absolute tolerance for the residual
        maxiter: maximum number of iterations to perform before aborting. Defaults to the system size.
            Note that the current implementation always performs iterations in pairs, and as a result may exceed the specified maximum number of iterations by one.
        M: optional left-preconditioner, ideally chosen such that ``M A`` is close to identity.
        callback: function to be called every `check_every` iteration with the current iteration number, residual and tolerance
        check_every: number of iterations every which to call `callback`, check the residual against the tolerance and possibility terminate the algorithm.
        use_cuda_graph: If true and when run on a CUDA device, capture the solver iteration as a CUDA graph for reduced launch overhead.
          The linear operator and preconditioner must only perform graph-friendly operations.

    Returns:
        Tuple (final iteration number, residual norm, absolute tolerance)

    If both `tol` and `atol` are provided, the absolute tolerance used as the termination criterion for the residual norm is ``max(atol, tol * norm(b))``.
    """

    A = aslinearoperator(A)
    M = aslinearoperator(M)

    if maxiter == 0:
        maxiter = A.shape[0]

    r, r_norm_sq, atol = _initialize_residual_and_tolerance(A, b, x, tol=tol, atol=atol)

    device = A.device
    scalar_dtype = wp.types.type_scalar_type(A.dtype)

    # Notations below follow roughly pseudo-code from https://en.wikipedia.org/wiki/Conjugate_residual_method
    # with z := M^-1 r and y := M^-1 Ap

    # z = M r
    if M is None:
        z = r
    else:
        z = wp.zeros_like(r)
        M.matvec(r, z, z, alpha=1.0, beta=0.0)

    Az = wp.zeros_like(b)
    A.matvec(z, Az, Az, alpha=1, beta=0)

    p = wp.clone(z)
    Ap = wp.clone(Az)

    if M is None:
        y = Ap
    else:
        y = wp.zeros_like(Ap)

    zAz_old = wp.empty(n=1, dtype=scalar_dtype, device=device)
    zAz_new = wp.empty(n=1, dtype=scalar_dtype, device=device)
    y_Ap = wp.empty(n=1, dtype=scalar_dtype, device=device)

    array_inner(z, Az, out=zAz_new)

    def do_iteration(atol_sq, rr, zAz_old, zAz_new):
        if M is not None:
            M.matvec(Ap, y, y, alpha=1.0, beta=0.0)
        array_inner(Ap, y, out=y_Ap)

        if M is None:
            # In non-preconditioned case, first kernel is same as CG
            wp.launch(
                kernel=_cg_kernel_1,
                dim=x.shape[0],
                device=device,
                inputs=[atol_sq, rr, zAz_old, y_Ap, x, r, p, Ap],
            )
        else:
            # In preconditioned case, we have one more vector to update
            wp.launch(
                kernel=_cr_kernel_1,
                dim=x.shape[0],
                device=device,
                inputs=[atol_sq, rr, zAz_old, y_Ap, x, r, z, p, Ap, y],
            )

        array_inner(r, r, out=rr)

        A.matvec(z, Az, Az, alpha=1, beta=0)
        array_inner(z, Az, out=zAz_new)

        # beta = rz_new / rz_old
        wp.launch(
            kernel=_cr_kernel_2,
            dim=z.shape[0],
            device=device,
            inputs=[atol_sq, rr, zAz_old, zAz_new, z, p, Az, Ap],
        )

    # We do iterations by pairs, switching old and new residual norm buffers for each odd-even couple
    def do_odd_even_cycle(atol_sq: float):
        do_iteration(atol_sq, r_norm_sq, zAz_new, zAz_old)
        do_iteration(atol_sq, r_norm_sq, zAz_old, zAz_new)

    return _run_solver_loop(
        do_odd_even_cycle,
        cycle_size=2,
        r_norm_sq=r_norm_sq,
        maxiter=maxiter,
        atol=atol,
        callback=callback,
        check_every=check_every,
        use_cuda_graph=use_cuda_graph,
        device=device,
    )


def _get_dtype_epsilon(dtype):
    if dtype == wp.float64:
        return 1.0e-16
    elif dtype == wp.float16:
        return 1.0e-4

    return 1.0e-8


def _get_absolute_tolerance(dtype, tol, atol, lhs_norm):
    eps_tol = _get_dtype_epsilon(dtype)
    default_tol = eps_tol ** (3 / 4)
    min_tol = eps_tol ** (9 / 4)

    if tol is None and atol is None:
        tol = atol = default_tol
    elif tol is None:
        tol = atol
    elif atol is None:
        atol = tol

    return max(tol * lhs_norm, atol, min_tol)


def _initialize_residual_and_tolerance(
    A: LinearOperator, b: wp.array, x: wp.array, tol: float, atol: float
):
    scalar_dtype = wp.types.type_scalar_type(A.dtype)
    device = A.device

    # Buffer for storing square norm or residual
    r_norm_sq = wp.empty(n=1, dtype=scalar_dtype, device=device, pinned=device.is_cuda)

    # Compute b norm to define absolute tolerance
    array_inner(b, b, out=r_norm_sq)
    atol = _get_absolute_tolerance(scalar_dtype, tol, atol, sqrt(r_norm_sq.numpy()[0]))

    # Residual r = b - Ax
    r = wp.empty_like(b)
    A.matvec(x, b, r, alpha=-1.0, beta=1.0)

    array_inner(r, r, out=r_norm_sq)

    return r, r_norm_sq, atol


def _run_solver_loop(
    do_cycle: Callable[[float], None],
    cycle_size: int,
    r_norm_sq: wp.array,
    maxiter: int,
    atol: float,
    callback: Callable,
    check_every: int,
    use_cuda_graph: bool,
    device,
):
    atol_sq = atol * atol

    cur_iter = 0

    err_sq = r_norm_sq.numpy()[0]
    err = sqrt(err_sq)
    if callback is not None:
        callback(cur_iter, err, atol)

    if err_sq <= atol_sq:
        return cur_iter, err, atol

    graph = None

    while True:
        # Do not do graph capture at first iteration -- modules may not be loaded yet
        if device.is_cuda and use_cuda_graph and cur_iter > 0:
            if graph is None:
                wp.capture_begin(device, force_module_load=False)
                try:
                    do_cycle(atol_sq)
                finally:
                    graph = wp.capture_end(device)
            wp.capture_launch(graph)
        else:
            do_cycle(atol_sq)

        cur_iter += cycle_size

        if cur_iter >= maxiter:
            break

        if (cur_iter % check_every) < cycle_size:
            err_sq = r_norm_sq.numpy()[0]

            if err_sq <= atol_sq:
                break

            if callback is not None:
                callback(cur_iter, sqrt(err_sq), atol)

    err_sq = r_norm_sq.numpy()[0]
    err = sqrt(err_sq)
    if callback is not None:
        callback(cur_iter, err, atol)

    return cur_iter, err, atol


@wp.kernel
def _cg_kernel_1(
    tol: Any,
    resid: wp.array(dtype=Any),
    rz_old: wp.array(dtype=Any),
    p_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
):
    i = wp.tid()

    alpha = wp.where(resid[0] > tol, rz_old[0] / p_Ap[0], rz_old.dtype(0.0))

    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]


@wp.kernel
def _cr_kernel_1(
    tol: Any,
    resid: wp.array(dtype=Any),
    zAz_old: wp.array(dtype=Any),
    y_Ap: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    r: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
    y: wp.array(dtype=Any),
):
    i = wp.tid()

    alpha = wp.where(
        resid[0] > tol and y_Ap[0] > 0.0, zAz_old[0] / y_Ap[0], zAz_old.dtype(0.0)
    )

    x[i] = x[i] + alpha * p[i]
    r[i] = r[i] - alpha * Ap[i]
    z[i] = z[i] - alpha * y[i]


@wp.kernel
def _cr_kernel_2(
    tol: Any,
    resid: wp.array(dtype=Any),
    zAz_old: wp.array(dtype=Any),
    zAz_new: wp.array(dtype=Any),
    z: wp.array(dtype=Any),
    p: wp.array(dtype=Any),
    Az: wp.array(dtype=Any),
    Ap: wp.array(dtype=Any),
):
    #    p = r + (rz_new / rz_old) * p;
    i = wp.tid()

    beta = wp.where(
        resid[0] > tol and zAz_old[0] > 0.0, zAz_new[0] / zAz_old[0], zAz_old.dtype(0.0)
    )

    p[i] = z[i] + beta * p[i]
    Ap[i] = Az[i] + beta * Ap[i]
