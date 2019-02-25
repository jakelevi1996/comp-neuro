import numpy as np
import plotting as p
from time import sleep

def tapped_delay_line(stim_mem): return stim_mem.copy()
def boxcar(stim_mem): return np.cumsum(stim_mem)

# Define time
T = np.arange(0, 25.5, 0.5)
# Define stimulus (IE state)
s = np.zeros(T.shape)
s[T == 10] = 1.0
# Define reward
r = 0.5 * np.exp(-0.5 * np.square(T - 20))


def dopamine_sim(
    feature_update=tapped_delay_line, gamma=1.0, epsilon=0.2, N_trials=201,
    mem_size=25
):
    # Initialise state value functions
    V = np.zeros([N_trials, T.size])
    # Initialise TD
    TD = np.zeros([N_trials, T.size])
    # Initialise TD
    learning_error = np.zeros([N_trials, T.size])
    # Initialise weights
    w = np.zeros(mem_size)
    # w = np.random.normal(mem_size)

    # Loop through trials
    for trial in range(N_trials):
        # Reset stimulus memory, features and old reward
        stim_mem, phi, r_old = np.zeros(mem_size), np.zeros(mem_size), 0
        # Iterate through time
        for t in range(T.size-1):
            # Update stimulus memory and features
            phi_old = phi.copy()
            stim_mem[1:] = stim_mem[:-1]
            stim_mem[0] = s[t]
            phi = feature_update(stim_mem)
            # Store new state-value, TD and learning error
            V[trial, t] = phi.dot(w)
            TD[trial, t] = gamma * V[trial, t] - phi_old.dot(w)
            learning_error[trial, t] = r_old + TD[trial, t]
            # Update weights
            w += epsilon * learning_error[trial, t] * phi_old
            # Store old reward
            r_old = r[t]
    print(w)
    
    return V, TD, learning_error


def q1():
    p.plot_stimulus_and_reward(T, s, r)
    V, TD, learning_error = dopamine_sim()
    p.plot_value_td_learning_error(T, V, TD, learning_error)

def q2():
    p.plot_stimulus_and_reward(T, s, r)
    V, TD, learning_error = dopamine_sim(
        feature_update=boxcar, epsilon=0.01,
        # N_trials=1001
    )
    p.plot_value_td_learning_error(
        T, V, TD, learning_error, filename="boxcar output signals",
        title="Output signals for boxcar TD-learning"
        # N_trials=1001, every_nth_trial=50
    )



if __name__ == "__main__":
    q1()
    q2()