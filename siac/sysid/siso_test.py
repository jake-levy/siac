from jax import numpy as jp
from siac.sysid import siso
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
step_amp = jp.array([1])
num_freqs = 2
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
C = jp.array([[1, 0]])

# Generate data
x_init = jp.zeros((2,))
step_input = inputs.step(step_amp, T, dt)
sine_input = inputs.sinusoidal(input_amps, input_freqs, T, dt)
us = step_input + sine_input
ys = integrate.euler_LTI(A, B, C, us, x_init, dt)
fig, ax = plt.subplots()
ax.plot(us)
ax.plot(ys)
plt.savefig("out.png")

# Hyperparameters
lam = jp.array([1, 1])  # LP filter
gamma = 1 * jp.eye(1)  # Learning rate
a_hat_init = jp.zeros((2,))
b_hat_init = jp.zeros((2,))

# Perform estimation (assuming m is known)
a_hat, a_hats = siso.estimate_a(us, ys, a_hat_init, b, gamma, lam, dt)
print("================ m is known (estimating 2 parameters) ================")
print("Gound truth:")
print(a)
print("Estimated:")
print(a_hat*m)
fig, ax = plt.subplots()
ax.plot(a_hats[:, 0]*m)
ax.plot(a_hats[:, 1]*m)
plt.savefig("est.png")

# Perform estimation (assuming m is unknown)
theta_hat, theta_hats = siso.estimate_ab(us, ys, a_hat_init, b_hat_init, gamma, lam, dt)
print("================ m is unknown (estimating 4 parameters) ================")
print("Gound truth:")
print(a)
print(b)
print("Estimated:")
print(theta_hat)
fig, ax = plt.subplots()
# ax.plot(theta_hats)
for i in range(theta_hats.shape[1]):
    ax.plot(theta_hats[:, i])
plt.savefig("est_m_unk.png")
