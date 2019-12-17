from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt

m_c = 1. # mass of cart in kg
m_p = .01  # mass of pole in kg
l = 0.25 # length of pole in m
g = 9.81 # m/s^2

def dynamics(y, t, f):
    assert(y.shape == (5,))
    x, v, theta, omega, prev_f = y

    sin = np.sin(theta)
    cos = np.cos(theta)
    alpha = m_c + m_p*sin**2
    beta = l*omega**2
    u = f(t) + prev_f

    dx_dt = v
    dv_dt = (u + m_p*sin*(beta + g*cos)) / alpha
    dtheta_dt = omega
    domega_dt = (-u*cos - m_p*beta*cos*sin - (m_c + m_p)*g*sin) / l / alpha
    dprev_f_dt = (f(t) - prev_f) / 0.01

    return [dx_dt, dv_dt, dtheta_dt, domega_dt, dprev_f_dt]

# horizontal position/velocity, angular pos/vel, previous control
# We include previous control so we can penalize for change in control inputs
initial_state = np.array([[0., 0., 0., 0., 0.]]).T
# So then actions are *changes* in actions
target_state_action = np.array([[0., 0., np.pi, 0., 0., 0.]]).T

# Reward shaping: angular position is most important
C = np.diag([0.01, 0.01, 1., 0.01, 0.00001, 0.1])
c = - np.dot(C, target_state_action)

zero_controller = (np.zeros_like(initial_state.T), np.zeros((1,1)))

def test_simulate_dynamics():
    y0 = initial_state.reshape(-1)
    t = np.linspace(0, 20, 1000)
    Omega = 1 # rad/s, control frequency
    f = lambda t: np.sin(Omega*t)

    y = odeint(dynamics, y0, t, args=(f,))
    xs, vs, thetas, omegas = zip(*y)
    thetas = np.array(thetas) * 180 / np.pi
    omegas = np.array(omegas) * 180 / np.pi

    # These graphs should show periodic but rather bumpy behavior
    plt.plot(t, xs); plt.savefig('cartpole_test_xs.png'); plt.clf()
    plt.plot(t, vs); plt.savefig('cartpole_test_vs.png'); plt.clf()
    plt.plot(t, thetas); plt.savefig('cartpole_test_thetas.png'); plt.clf()
    plt.plot(t, omegas); plt.savefig('cartpole_test_omegas.png'); plt.clf()

if __name__ == "__main__":
    test_simulate_dynamics()
