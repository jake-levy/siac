from jax import numpy as jp
import jax


def step(amps: jp.ndarray, T: float, dt: float) -> jp.ndarray:
    """Generate a step input with m channels.

    Args:
    - amps: amplitudes of the steps. Shape (m,)
    - T: time horizon
    - dt: time step

    Returns:
    - us: the input signal. Shape (T/dt, m)

    """
    us = amps * jp.ones((int(T / dt), amps.shape[0]))
    return us


def sinusoidal(amps: jp.ndarray, freqs: jp.ndarray,
               T: float, dt: float) -> jp.ndarray:
    """Generate a composition of sinusoidal inputs with m channels and d
    sinusoids.

    Args:
    - amps: amplitudes of the sinusoids. Shape (m, d)
    - freqs: frequencies of the sinusoids. Shape (m, d)
    - T: time horizon
    - dt: time step

    Returns:
    - us: the input signal. Shape (T/dt, m)

    """
    ts = jp.arange(0, T, dt)

    def f(unused_carry, t):
        return (), jp.sum(amps * jp.sin(2 * jp.pi * freqs * t), axis=1)

    us = jax.lax.scan(f, (), ts)[1]
    return us
