from jax import numpy as jp
import jax


def generate_output(us: jp.ndarray, theta_star: jp.ndarray) -> jp.ndarray:
    def f(unused_carry, u):
        return (), theta_star @ u

    ys = jax.lax.scan(f, (), us)[1]
    return ys


def estimate(us: jp.ndarray, ys: jp.ndarray, theta_hat_init: jp.ndarray,
             gamma: jp.ndarray, dt: float):

    def f(theta_hat, in_element):
        u, y = in_element
        y_hat = theta_hat @ u
        e_y = y_hat - y
        theta_hat_dot = -gamma @ jp.outer(e_y, u)
        theta_hat = theta_hat + theta_hat_dot * dt  # Euler integration
        return theta_hat, theta_hat

    theta_hat, theta_hats = jax.lax.scan(f, theta_hat_init, (us, ys))

    return theta_hat, theta_hats
