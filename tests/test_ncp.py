import numpy as np
import pytest
import warp as wp
from axion.ncp import scaled_fisher_burmeister_derivatives
from axion.ncp import scaled_fisher_burmeister_evaluate


# This file should test the functionality of the Scaled Fisher-Burmeister function, which is the NCP function used in the NSN engine.
# First test is the evaluation of the function itself.
# When the NCP function returns zero, the following should be true:
# 1. If the first argument is zero, the second argument should be larger than zero.
# 2. If the second argument is zero, the first argument should be larger than zero.


# Secondly the derivatives should be tested. The derivatives should be tested against finite differences.

ALPHA = 1.0
BETA = 1.0
EPSILON = 1e-12


@wp.kernel
def eval_wrapper(
    # inputs:
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    alpha: float,
    beta: float,
    epsilon: float,
    # outputs:
    result: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    result[i] = scaled_fisher_burmeister_evaluate(a[i], b[i], alpha, beta, epsilon)


@wp.kernel
def derivative_wrapper(
    # inputs:
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    alpha: float,
    beta: float,
    epsilon: float,
    # outputs:
    da: wp.array(dtype=wp.float32),
    db: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    da_val, db_val = scaled_fisher_burmeister_derivatives(
        a[i], b[i], alpha, beta, epsilon
    )
    da[i] = da_val
    db[i] = db_val


def test_scaled_fisher_burmeister_evaluate():
    wp.init()

    a_values = np.random.uniform(0.0, 10.0, size=100).astype(np.float32)
    b_values = np.random.uniform(0.0, 10.0, size=100).astype(np.float32)

    # Generate binary mask of where the a should be zero and b should remain positive
    mask = np.random.choice([True, False], size=100, p=[0.5, 0.5])

    # The mask determines which elements of a should be zero and b should remain positive
    a_values[mask] = 0.0
    b_values[~mask] = 0.0

    # Run the evaluation kernel
    a_wp = wp.array(a_values, dtype=wp.float32)
    b_wp = wp.array(b_values, dtype=wp.float32)
    result_wp = wp.empty_like(a_wp)
    wp.launch(
        eval_wrapper,
        dim=len(a_values),
        inputs=[a_wp, b_wp, ALPHA, BETA, EPSILON],
        outputs=[result_wp],
    )
    result = result_wp.numpy()
    # print("Result:", result)

    assert np.all(
        result < 1e-5
    ), "NCP function evaluation returned non-zero values unexpectedly."


def test_scaled_fisher_burmeister_derivatives():
    wp.init()

    # Generate random input values for a and b
    a_values = np.random.uniform(-10.0, 10.0, size=100).astype(np.float32)
    b_values = np.random.uniform(-10.0, 10.0, size=100).astype(np.float32)

    # Warp arrays for inputs and outputs
    a_wp = wp.array(a_values, dtype=wp.float32)
    b_wp = wp.array(b_values, dtype=wp.float32)
    da_wp = wp.empty_like(a_wp)
    db_wp = wp.empty_like(b_wp)

    # Launch kernel to compute analytic derivatives
    wp.launch(
        derivative_wrapper,
        dim=len(a_values),
        inputs=[a_wp, b_wp, ALPHA, BETA, EPSILON],
        outputs=[da_wp, db_wp],
    )

    # Retrieve analytic derivatives
    da_analytic = da_wp.numpy()
    db_analytic = db_wp.numpy()

    # Compute finite differences for numeric derivatives
    delta = 1e-3

    # First, compute the base function values
    fb_wp = wp.empty_like(a_wp)
    wp.launch(
        eval_wrapper,
        dim=len(a_values),
        inputs=[a_wp, b_wp, ALPHA, BETA, EPSILON],
        outputs=[fb_wp],
    )
    fb = fb_wp.numpy()

    # Perturb only a to compute da_numeric
    perturbed_a_values = a_values + delta
    perturbed_a_wp = wp.array(perturbed_a_values, dtype=wp.float32)
    perturbed_fb_wp = wp.empty_like(a_wp)

    wp.launch(
        eval_wrapper,
        dim=len(a_values),
        inputs=[perturbed_a_wp, b_wp, ALPHA, BETA, EPSILON],
        outputs=[perturbed_fb_wp],
    )
    perturbed_fb = perturbed_fb_wp.numpy()
    da_numeric = (perturbed_fb - fb) / delta

    # Perturb only b to compute db_numeric
    perturbed_b_values = b_values + delta
    perturbed_b_wp = wp.array(perturbed_b_values, dtype=wp.float32)

    wp.launch(
        eval_wrapper,
        dim=len(a_values),
        inputs=[a_wp, perturbed_b_wp, ALPHA, BETA, EPSILON],
        outputs=[perturbed_fb_wp],
    )
    perturbed_fb = perturbed_fb_wp.numpy()
    db_numeric = (perturbed_fb - fb) / delta

    ## Compute absolute differences for detailed comparison
    # da_diff = np.abs(da_analytic - da_numeric)
    # db_diff = np.abs(db_analytic - db_numeric)
    # print("Max da difference:", np.max(da_diff))
    # print("Max db difference:", np.max(db_diff))

    # Verify results with assertions
    assert np.allclose(
        da_analytic, da_numeric, atol=1e-2
    ), "Analytic derivative da does not match numeric derivative."
    assert np.allclose(
        db_analytic, db_numeric, atol=1e-2
    ), "Analytic derivative db does not match numeric derivative."


if __name__ == "__main__":
    test_scaled_fisher_burmeister_evaluate()
    test_scaled_fisher_burmeister_derivatives()
