from siac.utils import inputs
from siac.ac import mrac
from jax import numpy as jp
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


# Parameters
T = 100
dt = 0.01
x_init = jp.zeros((2,))

# Reference model
A_m = jp.array([[0,   1],
                [-1, -1]])
B_m = jp.array([[0],
                [1]])
rs = inputs.step(jp.ones(1,), T, dt)

# Ground truth
A = jp.array([[0,   1],
              [1,   1]])
B = jp.array([[0],
              [1]])

# Do MRAC
K_hat_init = jp.zeros((1, 2))
L_hat_init = jp.zeros((1, 1))

xs, x_ms, K_hats, L_hats = mrac.mrac(A, B, A_m, B_m,
                                     x_init, rs,
                                     K_hat_init, L_hat_init, dt)

# plot results
fig, ax = plt.subplots()
ax.plot(xs[:, 0])
ax.plot(x_ms[:, 0])
plt.savefig("mrac.png")
