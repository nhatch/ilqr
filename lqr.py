import numpy as np

def evaluate_trajectory(initial_state, controls, dynamics, cost):
    state = initial_state
    states = []
    actions = []
    costs = []
    for t in range(len(controls)):
        states.append(state)
        Kt, kt = controls[t]
        Ct, ct = cost[t]
        action = np.dot(Kt, state) + kt
        actions.append(action)
        state_action = np.concatenate((state,action))
        costs.append(calc_cost(state_action, Ct, ct))
        Ft, ft = dynamics[t]
        state = np.dot(Ft, state_action) + ft
    return states, actions, costs

def evaluate_trajectory_iterative(prev_states, prev_actions, controls, real_dynamics, cost, alpha=1.0):
    state = prev_states[0]
    states = []
    actions = []
    costs = []
    for t in range(len(controls)):
        states.append(state)
        state_differential = state - prev_states[t]
        Kt, kt = controls[t]
        Ct, ct = cost[t]
        base_action = prev_actions[t] + np.dot(Kt, state_differential)
        # Per-time-step line search is incredibly slow?? And it doesn't work anyway???
        #action = action_line_search(base_action, state, real_dynamics, kt, Ct, ct)
        action = prev_actions[t] + np.dot(Kt, state_differential) + alpha * kt
        actions.append(action)
        costs.append(calc_cost(np.concatenate((state,action)), Ct, ct))
        state = real_dynamics(state, action)
    return states, actions, costs

def action_line_search(base_action, state, dynamics, kt, Ct, ct):
    def action(alpha):
        return base_action + alpha * kt
    def next_state_cost(alpha):
        next_state = dynamics(state, action(alpha))
        state_action = np.concatenate((next_state, [[0.0]]))
        return calc_cost(state_action, Ct, ct)
    base = next_state_cost(0.0)
    alpha = 1.0
    while next_state_cost(alpha) > base: # TODO "compute the amount of improvement you expect"?
        alpha /= 2
    return action(alpha)

def calc_cost(state_action, Ct, ct):
    return (0.5 * np.dot(np.dot(state_action.T, Ct), state_action)
            + np.dot(ct.T, state_action))[0][0]

def lqr(dynamics, cost):
    controllers = []
    S = dynamics[0][1].shape[0]
    Ct, ct = cost[-1]
    Qt, qt = Ct, ct
    for t in reversed(range(1,len(dynamics))):
        Kt = - Qt[S:,:S] / Qt[S:,S:]
        kt = - qt[S:] / Qt[S:,S:]
        controllers.insert(0, (Kt, kt))
        Vt = Qt[:S,:S] + np.dot(Qt[:S,S:], Kt) + np.dot(Kt.T, Qt[S:,:S]) + Qt[S:,S:]*np.dot(Kt.T, Kt)
        vt = qt[:S] + Qt[:S,S:] * kt + Kt.T * qt[S:] + Kt.T * Qt[S:,S:] * kt
        Ft, ft = dynamics[t-1]
        Ct, ct = cost[t-1]
        Qt = Ct + np.dot(np.dot(Ft.T, Vt), Ft)
        qt = ct + np.dot(np.dot(Ft.T, Vt), ft) + np.dot(Ft.T, vt)
    Kt = - Qt[S:,:S] / Qt[S:,S:]
    kt = - qt[S:] / Qt[S:,S:]
    controllers.insert(0, (Kt, kt))
    return controllers
