import warp as wp


@wp.func
def scaled_fisher_burmeister_evaluate(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32,
    beta: wp.float32,
    epsilon: wp.float32,
) -> wp.float32:
    """Warp kernel-side evaluation of the Scaled Fisher-Burmeister function."""
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0 + epsilon)
    return scaled_a + scaled_b - norm


@wp.func
def scaled_fisher_burmeister_derivatives(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32,
    beta: wp.float32,
    epsilon: wp.float32,
) -> tuple[wp.float32, wp.float32]:
    """Warp kernel-side derivatives of the Scaled Fisher-Burmeister function."""
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0 + epsilon)

    da = alpha * (1.0 - scaled_a / norm)
    db = beta * (1.0 - scaled_b / norm)

    return da, db


@wp.func
def scaled_fisher_burmeister(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32 = 1.0,
    beta: wp.float32 = 1.0,
    epsilon: wp.float32 = 1e-6,
) -> tuple[wp.float32, wp.float32, wp.float32]:
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0 + epsilon)

    value = scaled_a + scaled_b - norm

    dvalue_da = alpha * (1.0 - scaled_a / norm)
    dvalue_db = beta * (1.0 - scaled_b / norm)

    return value, dvalue_da, dvalue_db
