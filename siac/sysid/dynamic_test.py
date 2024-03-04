from jax import numpy as jp
from siac.sysid import dynamic
from siac.utils import inputs
from siac.utils import transforms
from siac.utils import integrate
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


# Parameters
T = 5000
dt = 0.001
num_freqs = 6
input_amps = 1 * jp.ones((1, num_freqs))
input_freqs = jp.expand_dims(jp.linspace(0.01, 0.2, num_freqs), axis=0)

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

# Perform estimation (parallel model)
gamma_a = .1
gamma_b = .01
A_hat_init = jp.zeros((2, 2))
B_hat_init = jp.zeros((2, 1))
x_hat, A_hat, B_hat, x_hats, A_hats, B_hats = dynamic.parallel(
    us, ys, A_hat_init, B_hat_init, gamma_a, gamma_b, dt)
print("================ Parallel ================")
print("Gound truth:")
print(A)
print(B)
print("Estimated:")
print(A_hat)
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

# Perform estimation (series-parallel model)
Gamma_a = 10 * jp.eye(2)
Gamma_b = 10 * jp.eye(2)
A_m = -1 * jp.eye(2)
A_hat_init = jp.zeros((2, 2))
B_hat_init = jp.zeros((2, 1))
x_hat, A_hat, B_hat, x_hats, A_hats, B_hats = dynamic.series_parallel(
    us, ys, A_hat_init, B_hat_init, A_m, Gamma_a, Gamma_b, dt)
print("================ Series-Parallel ================")
print("Gound truth:")
print(A)
print(B)
print("Estimated:")
print(A_hat)
print(B_hat)
fig, ax = plt.subplots()
ax.plot(ys[:, 0])
ax.plot(x_hats[:, 0])
plt.savefig("xs_sp.png")
fig, ax = plt.subplots()
ax.plot(A_hats[:, 0, 0])
ax.plot(A_hats[:, 0, 1])
ax.plot(A_hats[:, 1, 0])
ax.plot(A_hats[:, 1, 1])
plt.savefig("A_sp.png")
fig, ax = plt.subplots()
ax.plot(B_hats[:, 0, 0])
ax.plot(B_hats[:, 1, 0])
plt.savefig("B_sp.png")
