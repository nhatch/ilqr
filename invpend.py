from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt

m = 1 # kg
g = 9.81 # m/s^2
l = 1 # m
b = 1 # damping, no idea what units
I = m*l*l

# Inverse pendulum dynamics
def dynamics(y, t, f):
    assert(y.shape == (2,))
    theta, omega = y
    u = f(t)
    dtheta_dt = omega
    domega_dt = (u - b*omega - m*g*l*np.sin(theta))/I
    return [dtheta_dt, domega_dt]

initial_state = np.array([[0., 0.]]).T
target_state_action = np.array([[np.pi, 0., 0.]]).T

C = np.diag([1., .1, 0.0001])
#C = np.diag([1, 1, 0.1]) # TODO I get division by zero if some of these are zero; why?

c = - np.dot(C, target_state_action)

zero_controller = (np.zeros((1,2)), np.zeros((1,1)))

def test_simulate_dynamics():
    y0 = initial_state.reshape(-1)
    Omega = 5
    f = lambda t: np.sin(Omega * t)
    t = np.linspace(0,10,1000)

    y = odeint(dynamics, y0, t, args=(f,))
    thetas, omegas = zip(*y)

    # These graphs should show sine waves of slightly erratic amplitude
    plt.plot(t, thetas)
    plt.savefig('invpend_test_thetas.png')
    plt.clf()
    plt.plot(t, omegas)
    plt.savefig('invpend_test_omegas.png')
    plt.clf()
    plt.plot(thetas, omegas)
    plt.savefig('invpend_test_thetasvsomegas.png')
    plt.clf()

if __name__ == "__main__":
    test_simulate_dynamics()
