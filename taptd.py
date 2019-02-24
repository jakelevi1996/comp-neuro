import numpy as np
import plotting as p
from time import sleep

# Define time
T = np.arange(0, 25.5, 0.5)
# Define stimulus (IE state)
s = np.zeros(T.shape)
s[T == 10] = 1.0
# Define reward
r = 0.5 * np.exp(-0.5 * np.square(T - 20))

p.plot_stimulus_and_reward(T, s, r)

# Parameters for TD
gamma = 1.0
epsilon = 0.2
N_trials = 201
mem_size = 25

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
    # Reset features and old reward
    phi, r_old = np.zeros(mem_size), 0
    # Iterate through time
    for t in range(T.size):
        # Update features
        phi_old = phi.copy()
        phi[0], phi[1:] = s[t], phi[:-1]
        #  = 
        # Store new state-value
        V[trial, t] = phi.dot(w)
        # Store TD
        TD[trial, t] = gamma * V[trial, t] - phi_old.dot(w)
        # Store learning error
        learning_error[trial, t] = r_old + TD[trial, t]
        # Update weights
        w += epsilon * learning_error[trial, t] * phi_old
        # Store old reward
        r_old = r[t]

print(w)

p.plot_value_td_learning_error(T, V, TD, learning_error, N_trials=201, every_nth_trial=10)


        # if trial > 50:
        #     print(
        #         "*"*10, "Trial {}, t = {}".format(trial, t), "State:", phi,
        #         "Old reward = {:.5}, Value = {:.5}, TD = {:.5}, LE = {:.5}".format(
        #             r_old, V[trial, t], TD[trial, t], learning_error[trial, t]
        #         ), "Weights:", w, sep="\n"
        #     )
        #     sleep(0.5)