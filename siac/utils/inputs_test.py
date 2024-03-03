from siac.utils import inputs
from jax import numpy as jp

print(inputs.sinusoidal(jp.array([1, 2]), jp.array([1, 2]), 1, 0.1))
