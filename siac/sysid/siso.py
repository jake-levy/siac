from siac.sysid import static
from jax import numpy as jp
import jax


def estimate_a(us: jp.ndarray, ys: jp.ndarray, a_hat_init: jp.ndarray,
               b: jp.ndarray, gamma: jp.ndarray, lam: jp.ndarray,
               dt: float) -> jp.ndarray:

    n = lam.shape[0]
    A_c = jp.zeros((n, n))
    A_c = A_c.at[0].set(-lam)
    A_c = A_c.at[jp.arange(1, n), jp.arange(n-1)].set(1)
    B_c = jp.zeros((n,))
    B_c = B_c.at[0].set(1)

    def f(carry, in_element):
        phi1, phi2 = carry
        u, y = in_element

        phi1_dot = A_c @ phi1 + B_c * u
        phi2_dot = A_c @ phi2 - B_c * y

        # Euler integration
        phi1 = phi1 + phi1_dot * dt
        phi2 = phi2 + phi2_dot * dt

        input = phi2
        output = y + jp.dot(lam, phi2) - jp.dot(b, phi1)

        return (phi1, phi2), (input, output)

    phi1_init = jp.zeros((n,))
    phi2_init = jp.zeros((n,))
    _, (inputs, outputs) = jax.lax.scan(f, (phi1_init, phi2_init), (us, ys))
    a_hat, a_hats = static.estimate(inputs, outputs,
                                    jp.expand_dims(a_hat_init, axis=1).T,
                                    gamma, dt)
    return a_hat, a_hats

def estimate_ab(us: jp.ndarray, ys: jp.ndarray, a_hat_init: jp.ndarray,
               b_hat_init: jp.ndarray, gamma: jp.ndarray, lam: jp.ndarray,
               dt: float) -> jp.ndarray:

    n = lam.shape[0]
    A_c = jp.zeros((n, n))
    A_c = A_c.at[0].set(-lam)
    A_c = A_c.at[jp.arange(1, n), jp.arange(n-1)].set(1)
    B_c = jp.zeros((n,))
    B_c = B_c.at[0].set(1)

    def f(carry, in_element):
        phi1, phi2 = carry
        u, y = in_element

        phi1_dot = A_c @ phi1 + B_c * u
        phi2_dot = A_c @ phi2 - B_c * y

        # Euler integration
        phi1 = phi1 + phi1_dot * dt
        phi2 = phi2 + phi2_dot * dt

        input = jp.concatenate((phi1, phi2))
        output = y + jp.dot(lam, phi2) 

        return (phi1, phi2), (input, output)

    phi1_init = jp.zeros((n,))
    phi2_init = jp.zeros((n,))
    _, (inputs, outputs) = jax.lax.scan(f, (phi1_init, phi2_init), (us, ys))
    theta_hat, theta_hats = static.estimate(
        inputs, outputs,
        jp.expand_dims(jp.concatenate((a_hat_init, b_hat_init)), axis=1).T,
        gamma, dt)
    return theta_hat, theta_hats
