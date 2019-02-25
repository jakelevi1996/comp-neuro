import numpy as np
import plotting
from time import sleep

def tapped_delay_line(stim_mem): return stim_mem.copy()
def boxcar(stim_mem): return np.cumsum(stim_mem)

# Define time, stimulus (IE state) and reward
T = np.arange(0, 25.5, 0.5)
s = np.zeros(T.shape)
s[T == 10] = 1.0
r = 0.5 * np.exp(-0.5 * np.square(T - 20))


def dopamine_sim(
    feature_update=tapped_delay_line, gamma=1.0, epsilon=0.2, N_trials=201,
    mem_size=25, p=1.0
):
    # Initialise state value functions, TD, learning error, and rewarded trials
    V = np.zeros([N_trials, T.size])
    TD = np.zeros([N_trials, T.size])
    learning_error = np.zeros([N_trials, T.size])
    rewarded_trials = np.random.binomial(n=1, p=p, size=N_trials)
    # Initialise weights
    w = np.zeros(mem_size)
    # w = np.random.normal(mem_size)

    # Loop through trials
    for trial in range(N_trials):
        # Reset stimulus memory, features and old reward
        stim_mem, phi, r_old = np.zeros(mem_size), np.zeros(mem_size), 0
        # Determine if reward is presented
        if rewarded_trials[trial] == 1: reward = r
        else: reward = np.zeros(r.shape)
        # Iterate through time
        for t in range(T.size):
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
            r_old = reward[t]
    print(w)
    
    return V, TD, learning_error, rewarded_trials

def dopamine_activation_function(x, x_sat=0.27, alpha=6.0, beta=6.0):
    y = x.copy()
    y[x < 0] = y[x < 0] / alpha
    y[x > x_sat] = x_sat + (y[x > x_sat] - x_sat) / beta
    return y


def q3():
    plotting.plot_stimulus_and_reward(T, s, r, filename="q3a input signals")
    V, TD, learning_error, _ = dopamine_sim()
    plotting.plot_td_learning_signals(
        T, V, TD, learning_error, filename="q3b tapdl output signals",
        title="Output signals for tapped delay-line TD-learning"
    )

def q4():
    V, TD, learning_error, _ = dopamine_sim(
        feature_update=boxcar, epsilon=0.01
    )
    plotting.plot_td_learning_signals(
        T, V, TD, learning_error, filename="q4a boxcar output signals",
        title="Output signals for boxcar TD-learning"
    )

def q5():
    V, TD, learning_error, rewarded_trials = dopamine_sim(
        feature_update=boxcar, epsilon=0.01, N_trials=1000, p=0.5
    )
    plotting.plot_partial_reinforcements(
        T, V, TD, learning_error, rewarded_trials,
        "q5a partial reinforcements",
        title="Output signals using partial reinforcement"
    )
    dopamine_activity = dopamine_activation_function(learning_error)
    plotting.plot_dopamine_vs_learning_error(
        T, dopamine_activity, learning_error, "q5c dopamine time course"
    )

def q6():
    p_list = np.arange(0.0, 1.01, 0.1)
    dopamine_signals = []
    for p in p_list:
        print("p =", p)
        _, _, learning_error, _ = dopamine_sim(
            feature_update=boxcar, epsilon=0.01, N_trials=1500, p=p
        )
        d = dopamine_activation_function(learning_error)[-500:].mean(axis=0)
        dopamine_signals.append(d)
    plotting.plot_probabilistic_dopamine(
        T, dopamine_signals, p_list, "q6 probabilistic dopamine"
    )

def q7():
    p_list = np.arange(0.0, 1.01, 0.1)
    peak_dopamine_at_stimulus, peak_dopamine_at_reward = [], []

    for p in p_list:
        print("p =", p)
        _, _, learning_error, _ = dopamine_sim(
            feature_update=boxcar, epsilon=0.01, N_trials=1500, p=p
        )
        d = dopamine_activation_function(learning_error)[-500:].mean(axis=0)
        peak_dopamine_at_stimulus.append(max(d[T < 15]))
        peak_dopamine_at_reward.append(max(d[T > 15]))
    
    plotting.plot_dopamine_peaks(
        p_list, peak_dopamine_at_stimulus, peak_dopamine_at_reward,
        filename="q7 dopamine peaks"
    )




if __name__ == "__main__":
    # q3()
    # q4()
    # q5()
    # q6()
    q7()