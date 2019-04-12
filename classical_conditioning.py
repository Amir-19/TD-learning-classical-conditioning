#CSC implementation init

import numpy as np
from dynamic_plotter import *
from td import TD

def next_state(state, num_state):
    state_prime = (state + 1) % (num_state)
    return state_prime, stimuli(state_prime,num_state)

def feature_vector(state, num_state):
    fv = np.zeros(num_state)
    fv[state] = 1.0
    return fv

def stimuli(state, num_state):
    if state == (num_state - 1):
        return 1.0
    else:
        return 0.0

def experiment():

    plotting = True
    if plotting:
        d = DynamicPlot(window_x = 100, title = 'On-Policy Predictions', xlabel = 'Tim  e_Step', ylabel= 'Value')
        d.add_line('Prediction')
        d.add_line('State')

    # init problem
    num_state = 25

    alpha = 0.5
    lam = 0.95
    gamma = 0.97

    # init state, action, and time step
    state = 0
    t = 0

    # init the solution
    soul = TD(num_state)
    # TD lambda algorithm main loop
    while True:
        state_prime, stim = next_state(state, num_state)
        if state_prime == 0:
            soul.reset_et()
        else:
            delta = soul.update(feature_vector(state,num_state),stim,feature_vector(state_prime,num_state),alpha,gamma,gamma,lam)
        d.update(t, [soul.get_value(feature_vector(state,num_state)),0])
        state = state_prime
        t += 1


if __name__ == "__main__":
    experiment()
