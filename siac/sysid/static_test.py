from siac.sysid import static
from siac.utils.inputs import sinusoidal
from jax import numpy as jp
import jax


T = 10      # time horizon
dt = 0.01   # time step
n = 3       # input dim
m = 2       # output dim
seed = 1    # random seed

# Generate keys
key = jax.random.PRNGKey(seed)
key_theta, key_amp, key_freq = jax.random.split(key, 3)

# Generate a random ground truth parameter, theta_star
theta_star = jax.random.uniform(key_theta, minval=0, maxval=1, shape=(m, n))

# Generate a sinusoidal input
amps = jax.random.uniform(key_amp, minval=0.1, maxval=1, shape=(n,))
freqs = jax.random.uniform(key_freq, minval=0.1, maxval=1, shape=(n,))
us = sinusoidal(amps, freqs, T, dt)

# Generate the output
ys = static.generate_output(us, theta_star)

# Estimate the parameter
theta_hat_init = jp.zeros((m, n))
gamma = jp.eye(m)
theta_hat, theta_hats = static.estimate(us, ys, theta_hat_init, gamma, dt)
print(theta_hat)
print(theta_star)
