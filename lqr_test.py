import numpy as np
import matplotlib.pyplot as plt
import lqr
from cartpole import C, c, initial_state

horizon = 7.0 # Time horizon for trajectory optimization, in seconds
deltaT = 1e-3 # Time discretization
T = int(horizon / deltaT)

cost = [(C, c) for _ in range(T)]

# Sort of like cartpole, but with made-up simplified linear dynamics
F = np.array([
      [1, deltaT, 0, 0, 0],
      [0, 1, 0, 0, deltaT],
      [0, 0, 1, deltaT, 0],
      [deltaT, 0, 0, 1, 0]])
f = np.zeros((4,1))
dynamics = [(F, f) for _ in range(T)]

controls = lqr.lqr(dynamics, cost)
states, actions, costs = lqr.evaluate_trajectory(initial_state, controls, dynamics, cost)

x, v, theta, omega = zip(*states)
actions = np.array(actions).flatten()
t = np.linspace(0, horizon, T)

# These graphs should show reasonable behavior: we reach the target state after ~4sec
plt.plot(t, theta); plt.savefig('lqr_test_theta.png'); plt.clf()
plt.plot(t, actions); plt.savefig('lqr_test_actions.png'); plt.clf()
plt.plot(t, x); plt.savefig('lqr_test_x.png'); plt.clf()

# Some numerical checks that we found a good solution
target_cost = -0.5 * 100 * np.pi**2
assert(abs(costs[-1] - target_cost) < 1)
assert(sum(costs)/T < -416)
