import warp as wp


@wp.func
def scaled_fisher_burmeister(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32 = 1.0,
    beta: wp.float32 = 1.0,
):
    """
    Computes the Scaled Fisher-Burmeister NCP function value.
    Phi(a, b) = alpha*a + beta*b - sqrt((alpha*a)^2 + (beta*b)^2)
    """
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0)

    return scaled_a + scaled_b - norm


@wp.func
def scaled_fisher_burmeister_diff(
    a: wp.float32,
    b: wp.float32,
    alpha: wp.float32 = 1.0,
    beta: wp.float32 = 1.0,
):
    """
    Computes the Scaled Fisher-Burmeister NCP function value and its partial derivatives.
    Returns: (value, dvalue_da, dvalue_db)
    """
    scaled_a = alpha * a
    scaled_b = beta * b
    norm = wp.sqrt(scaled_a**2.0 + scaled_b**2.0)

    value = scaled_a + scaled_b - norm

    # Avoid division by zero
    if norm < 1e-5:
        return value, 0.0, 1.0

    dvalue_da = alpha * (1.0 - scaled_a / norm)
    dvalue_db = beta * (1.0 - scaled_b / norm)

    return value, dvalue_da, dvalue_db