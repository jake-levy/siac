from jax import numpy as jp
import jax


def euler_LTI(A: jp.ndarray, B: jp.ndarray, C: jp.ndarray,
              us: jp.ndarray, x_init: jp.ndarray,
              dt: float) -> jp.ndarray:
    """Euler integration of a linear time invariant system."""
    def f(carry, in_element):
        x = carry
        u = in_element
        y = C @ x
        x_dot = A @ x + B @ u
        x = x + x_dot * dt
        return x, y

    x, ys = jax.lax.scan(f, x_init, us)
    return ys
