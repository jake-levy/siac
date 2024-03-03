from jax import numpy as jp
import jax


def sinusoidal(amps: jp.ndarray, freqs: jp.ndarray,
               T: float, dt: float) -> jp.ndarray:
    """Generate a sinusoidal input. Shape (T*dt, m)"""
    ts = jp.arange(0, T, dt)

    def f(unused_carry, t):
        return (), amps * jp.sin(2 * jp.pi * freqs * t)

    us = jax.lax.scan(f, (), ts)[1]
    return us
