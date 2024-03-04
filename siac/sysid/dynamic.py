from jax import numpy as jp
import jax
import scipy.linalg as sla
import numpy as np


def parallel(us, xs, A_hat_init: jp.ndarray, B_hat_init: jp.ndarray,
             gamma_a: float, gamma_b: float, dt: float):

    def f(carry, in_element):

        x_hat, A_hat, B_hat = carry
        u, x = in_element
        e_x = x_hat - x

        # diff eqs
        x_hat_dot = A_hat @ x_hat + B_hat @ u
        A_hat_dot = -gamma_a * jp.outer(e_x, x_hat)
        B_hat_dot = -gamma_b * jp.outer(e_x, u)

        # Euler integration
        x_hat = x_hat + x_hat_dot * dt
        A_hat = A_hat + A_hat_dot * dt
        B_hat = B_hat + B_hat_dot * dt

        return (x_hat, A_hat, B_hat), (x_hat, A_hat, B_hat)

    x_hat_init = xs[0, :]
    (x_hat, A_hat, B_hat), (x_hats, A_hats, B_hats) = jax.lax.scan(
        f, (x_hat_init, A_hat_init, B_hat_init), (us, xs))

    return x_hat, A_hat, B_hat, x_hats, A_hats, B_hats


def series_parallel(us, xs, A_hat_init: jp.ndarray, B_hat_init: jp.ndarray,
                    A_m: jp.ndarray, Gamma_a: float, Gamma_b: float,
                    dt: float):

    Gamma_a_inv = jp.linalg.inv(Gamma_a)
    Gamma_b_inv = jp.linalg.inv(Gamma_b)

    n = A_hat_init.shape[0]
    P = sla.solve_continuous_lyapunov(np.array(A_m.T), -np.eye(n))
    P = jp.array(P)

    def f(carry, in_element):

        x_hat, A_hat, B_hat = carry
        u, x = in_element
        e_x = x_hat - x

        # diff eqs
        x_hat_dot = A_m @ e_x + A_hat @ x + B_hat @ u
        A_hat_dot = -Gamma_a_inv @ P @ jp.outer(e_x, x)
        B_hat_dot = -Gamma_b_inv @ P @ jp.outer(e_x, u)

        # Euler integration
        x_hat = x_hat + x_hat_dot * dt
        A_hat = A_hat + A_hat_dot * dt
        B_hat = B_hat + B_hat_dot * dt

        return (x_hat, A_hat, B_hat), (x_hat, A_hat, B_hat)

    x_hat_init = xs[0, :]
    (x_hat, A_hat, B_hat), (x_hats, A_hats, B_hats) = jax.lax.scan(
        f, (x_hat_init, A_hat_init, B_hat_init), (us, xs))

    return x_hat, A_hat, B_hat, x_hats, A_hats, B_hats
