from jax import numpy as jp
from siac.sysid import dynamic
from siac.utils import inputs
from siac.utils import transforms
from siac.utils import integrate
import matplotlib.pyplot as plt
import jax


# Parameters
T = 1000
dt = 0.01
num_freqs = 4
input_amps = jp.ones((1, num_freqs))
key = jax.random.PRNGKey(0)
input_freqs = jax.random.uniform(key, minval=1, maxval=10,
                                 shape=(1, num_freqs))

# Ground truth
m = 20
c = 0.1
k = 5
a = jp.array([c/m, k/m])
A = transforms.siso_a_coeffs_to_A(a)
b = jp.array([0, 1/m])
B = jp.expand_dims(b, axis=1)
C = jp.eye(2)  # => xs = ys

# Generate data
x_init = jp.zeros((2,))
us = inputs.sinusoidal(input_amps, input_freqs, T, dt)
ys = integrate.euler_LTI(A, B, C, us, x_init, dt)
fig, ax = plt.subplots()
ax.plot(ys[:, 0])
ax.plot(ys[:, 1])
plt.savefig("out.png")

# Hyperparameters
gamma_a = 1
gamma_b = 1
A_hat_init = jp.zeros((2, 2))
B_hat_init = jp.zeros((2, 1))

# Perform estimation
x_hat, A_hat, B_hat, x_hats, A_hats, B_hats = dynamic.parallel(
    us, ys, A_hat_init, B_hat_init, gamma_a, gamma_b, dt)
print(A)
print(A_hat)
print(B)
print(B_hat)
fig, ax = plt.subplots()
ax.plot(ys[:, 0])
ax.plot(x_hats[:, 0])
plt.savefig("xs.png")
fig, ax = plt.subplots()
ax.plot(A_hats[:, 0, 0])
ax.plot(A_hats[:, 0, 1])
ax.plot(A_hats[:, 1, 0])
ax.plot(A_hats[:, 1, 1])
plt.savefig("A.png")
fig, ax = plt.subplots()
ax.plot(B_hats[:, 0, 0])
ax.plot(B_hats[:, 1, 0])
plt.savefig("B.png")
