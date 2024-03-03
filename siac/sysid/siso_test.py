from jax import numpy as jp
from siac.sysid import siso
from siac.utils import inputs
from siac.utils import transforms
from siac.utils import integrate
import matplotlib.pyplot as plt


# Parameters
T = 100
dt = 0.01
input_amp = 1
input_freq = 1

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
us = 10 + inputs.sinusoidal(input_amp, input_freq, T, dt)
us = jp.expand_dims(us, axis=1)
ys = integrate.euler_LTI(A, B, C, us, x_init, dt)
fig, ax = plt.subplots()
# ax.plot(us)
ax.plot(ys)
plt.savefig("out.png")


# Hyperparameters
lam = jp.array([1, 1])  # LP filter
gamma = 1 * jp.eye(1)  # Learning rate
a_hat_init = jp.zeros((2,))
b_hat_init = jp.zeros((2,))

a_hat, a_hats = siso.estimate_a(us, ys, a_hat_init, b, gamma, lam, dt)
print(a_hat*m)
fig, ax = plt.subplots()
ax.plot(a_hats[:, 0]*m)
ax.plot(a_hats[:, 1]*m)
plt.savefig("est.png")

# theta_hat, theta_hats = siso.estimate_ab(us, ys, a_hat_init, b_hat_init, gamma, lam, dt)
# print(a)
# print(b)
# print(theta_hat)
# fig, ax = plt.subplots()
# # ax.plot(theta_hats)
# for i in range(theta_hats.shape[1]):
#     ax.plot(theta_hats[:, i])
# plt.savefig("est.png")