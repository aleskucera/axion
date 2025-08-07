import warp as wp


@wp.func
def scaled_fisher_burmeister_evaluate(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32,
    beta: wp.float32,
) -> wp.float32:
    """Warp kernel-side evaluation of the Scaled Fisher-Burmeister function."""
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0)
    return scaled_a + scaled_b - norm


@wp.func
def scaled_fisher_burmeister_derivatives(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32,
    beta: wp.float32,
):
    """Warp kernel-side derivatives of the Scaled Fisher-Burmeister function."""
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0)

    # Avoid division by zero
    if norm < 1e-6:
        return 1.0, 0.0

    da = alpha * (1.0 - scaled_a / norm)
    db = beta * (1.0 - scaled_b / norm)

    return da, db


@wp.func
def scaled_fisher_burmeister(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32 = 1.0,
    beta: wp.float32 = 1.0,
):
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0)

    value = scaled_a + scaled_b - norm

    # Avoid division by zero
    if norm < 1e-6:
        return value, 1.0, 0.0

    dvalue_da = alpha * (1.0 - scaled_a / norm)
    dvalue_db = beta * (1.0 - scaled_b / norm)

    return value, dvalue_da, dvalue_db
