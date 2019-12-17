import numpy as np
import cartpole
import lqr
import stepper
#import invpend as env
import cartpole as env
from matplotlib import pyplot as plt

def linear_dynamic(dynamics, state, action):
    # Estimate the gradient based on numerical differentiation.
    # Question: Is it possible to do this analytically, by manipulating the differential equation?
    x = np.concatenate((state, action))
    def func(state_action):
        return dynamics(state_action[:-1], state_action[-1:])
    base = func(x)
    epsilon = 1e-8
    dxs = [func(x+dx.reshape(-1,1)) - base for dx in np.identity(len(x))*epsilon]
    F = np.concatenate(dxs, 1) / epsilon
    f = np.zeros_like(base) # Question: Should this just be base?
                            # No: These dynamics return the *difference* from the state
                            # at this time step on the previous trajectory.
    return (F, f)

def recenter_cost(cost, states, actions):
    def recenter_cost(C, c, s, a):
        # We can pick larger values for epsilon to encourage LQR not to deviate from the
        # current trajectory. This is important because the linear approximation of the
        # dynamics is valid only locally.
        epsilon = 0.0
        new_C = C + epsilon * np.identity(C.shape[0])
        return C, c + np.dot(C, np.concatenate((s,a)))
    return [recenter_cost(C, c, s, a) for (C, c), (s, a) in zip(cost, zip(states, actions))]

def linear_dynamics(dynamics, states, actions):
    return [linear_dynamic(dynamics, state, action) for state, action in zip(states, actions)]

def save_trajectory(states, actions):
    # TODO move this code into the environment file so I don't have to keep commenting lines
    # when I switch environments.
    x, v, theta, omega, actions = zip(*states)
    #theta, omega = zip(*states)
    #actions = np.array(actions).flatten()
    t = np.linspace(0, horizon, T)

    plt.plot(t, actions); plt.savefig('ilqr_test_actions.png'); plt.clf()
    plt.plot(t, theta); plt.savefig('ilqr_test_theta.png'); plt.clf()
    plt.plot(t, omega); plt.savefig('ilqr_test_omega.png'); plt.clf()
    plt.plot(t, x); plt.savefig('ilqr_test_x.png'); plt.clf()
    plt.plot(t, v); plt.savefig('ilqr_test_v.png'); plt.clf()

def state_diff(states, lqr_states):
    diffs = []
    for s1, s2 in zip(states, lqr_states):
        diffs.append(((s1-s2)**2).mean())
    print("approximation error:")
    for i in [0,1,10,100,-2,-1]:
        print("  {} ({:.2f} {:.2f}, {:.2f} {:.2f})".format(diffs[i], lqr_states[i][0,0], lqr_states[i][1,0], states[i][0,0], states[i][1,0]))


def ilqr(initial_state, dynamics, cost, initial_controllers):
    action_dim, state_dim = env.zero_controller[0].shape
    states = [np.zeros((state_dim, 1))]*T
    actions = [np.zeros((action_dim, 1))]*T
    states, actions, costs = lqr.evaluate_trajectory_iterative(states, actions, initial_controllers, dynamics, cost)
    c = sum(costs) / T
    dc = 1
    epsilon = 1e-5
    iters = 0
    while dc > epsilon:
        iters += 1
        approx_dynamics = linear_dynamics(dynamics, states, actions)
        approx_cost = recenter_cost(cost, states, actions)
        controls = lqr.lqr(approx_dynamics, approx_cost)
        lqr_states, _, lqr_costs = lqr.evaluate_trajectory(initial_state, controls, approx_dynamics, cost)
        #print("LQR predicted cost:", sum(lqr_costs) / T, "and final state:", lqr_states[-1][0])
        alpha = 1.0
        while True:
            next_states, next_actions, costs = lqr.evaluate_trajectory_iterative(states, actions, controls, dynamics, cost, alpha=alpha)
            next_c = sum(costs) / T
            # This is a *global* version of line search (as opposed to choosing alpha
            # per time step).
            # Choosing alpha per time step would be a more complicated implementation, and
            # probably would not work properly (it would not account for downstream effects).
            if next_c < c:
                break
            alpha /= 2
        states = next_states
        actions = next_actions
        #state_diff(states, lqr_states)
        dc = np.abs(c - next_c)
        c = next_c
        print(iters, ":", next_c, dc)
        save_trajectory(states, actions)
    return states, actions, costs

horizon = 8 # Time horizon for trajectory optimization, in seconds
deltaT = 1e-2 # Time discretization
T = int(horizon / deltaT)
def dyno(state, action):
    return stepper.step(state.reshape(-1), action[0,0], deltaT, env.dynamics).reshape(-1, 1)
cost = [(env.C, env.c) for _ in range(T)]
initial_controllers = [env.zero_controller]*T

states, actions, costs = ilqr(env.initial_state, dyno, cost, initial_controllers)
save_trajectory(states, actions)

