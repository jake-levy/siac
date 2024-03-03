from jax import numpy as jp


def siso_a_coeffs_to_A(a):
    n = len(a)
    A = jp.zeros((n, n))
    A = A.at[jp.arange(n-1), jp.arange(1, n)].set(1)
    A = A.at[-1].set(-jp.flip(a))
    return A

