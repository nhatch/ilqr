import numpy as np
from scipy.integrate import odeint
import numbers

def step(state, action, deltaT, dynamics):
    t = [0.0, deltaT]
    # Question: should we allow the control to vary during this time interval?
    # Even a linear controller might do that, varying the control as the state changes.
    # But robotic actuators probably can only do discrete time?
    # We would need to change the `step` interface to accept a control function,
    # rather than just a constant `action`.
    assert(isinstance(action, numbers.Number))
    f = lambda _: action
    y = odeint(dynamics, state, t, args=(f,))
    return np.array(y[1])
