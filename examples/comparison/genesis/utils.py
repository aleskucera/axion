"""Minimal stub — only assert_allclose is needed from the Genesis test utils."""
import numpy as np
import torch


def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=None):
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if tol is not None:
        atol = rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    np.testing.assert_allclose(to_numpy(actual), to_numpy(desired),
                                atol=atol, rtol=rtol, err_msg=err_msg)
