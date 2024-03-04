from jax import numpy as jp
import scipy.linalg as sla
import numpy as np
import jax


def mrac(A: jp.ndarray, B: jp.ndarray,
         A_m: jp.ndarray, B_m: jp.ndarray,
         x_init: jp.ndarray, rs: jp.ndarray,
         K_hat_init: jp.ndarray, L_hat_init: jp.ndarray,
         dt: float):

    n = A_m.shape[0]
    P = sla.solve_continuous_lyapunov(np.array(A_m.T), -np.eye(n))
    P = jp.array(P)

    def f(carry, in_element):

        x, x_m, K_hat, L_hat = carry
        r = in_element
        e = x - x_m

        u = -K_hat @ x + L_hat @ r
        x_dot = A @ x + B @ u
        x_m_dot = A_m @ x_m + B_m @ r
        K_hat_dot = B_m.T @ P @ jp.outer(e, x)
        L_hat_dot = -B_m.T @ P @ jp.outer(e, r)

        # Euler integration
        x = x + x_dot * dt
        x_m = x_m + x_m_dot * dt
        K_hat = K_hat + K_hat_dot * dt
        L_hat = L_hat + L_hat_dot * dt

        return (x, x_m, K_hat, L_hat), (x, x_m, K_hat, L_hat)

    x_m_init = x_init
    _, (xs, x_ms, K_hats, L_hats) = jax.lax.scan(
        f, (x_init, x_m_init, K_hat_init, L_hat_init), rs)

    return xs, x_ms, K_hats, L_hats
