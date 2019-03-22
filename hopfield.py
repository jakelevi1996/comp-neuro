import numpy as np
import matplotlib.pyplot as plt
import plotting
from scipy.stats import norm
from time import time

def theoretical_error_prob(n_neurons, n_memories):
    return norm.cdf(-np.sqrt((n_neurons - 1) / (2 * n_memories - 1)))

def gen_random_memories(n_neurons, n_memories, p=0.5):
    return np.random.binomial(1, p, size=[n_memories, n_neurons])

def step(x): return 1.0 if x >= 0.0 else 0.0

def relu(x): return np.where(x > 0.0, x, 0.0)

def simulate_hopfield_net(memories, initial_state, random_order=True):
    # Check arrays are the right size
    n_neurons = memories.shape[1]
    assert initial_state.shape == (n_neurons, )
    # Initialise weights
    weights = (memories - 0.5).T.dot(memories - 0.5)
    i = np.arange(n_neurons)
    weights[i, i] = 0.0
    # Initialise states
    old_state, new_state = None, initial_state.copy()
    
    # Loop until convergence:
    while np.any(old_state != new_state):
        # Store old state
        old_state = new_state.copy()
        # Choose order in which to update neuron states
        if random_order: neuron_order = np.random.permutation(n_neurons)
        else: neuron_order = range(n_neurons)
        # Update neuron states asynchronously
        for i in neuron_order: new_state[i] = step(weights[i].dot(new_state))
    
    return new_state


def q1():
    n_neurons = 100
    n_memories = np.logspace(0, 3, 100)
    error_prob = theoretical_error_prob(n_neurons, n_memories)
    
    # Plot results
    plt.figure(figsize=[8, 6])
    plt.semilogx(n_memories, error_prob, "r")
    plt.xlabel("Number of input memories")
    plt.ylabel("Probability of incorrectly flipping a bit")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("q1 theoretical error probability N = 100")
    plt.close()


def q2():
    n_memories = np.logspace(0, 3, 100)
    error_prob_100 = theoretical_error_prob(100, n_memories)
    error_prob_1000 = theoretical_error_prob(1000, n_memories)
    
    # Plot results
    plt.figure(figsize=[8, 6])
    plt.semilogx(n_memories, error_prob_100, "r")
    plt.semilogx(n_memories, error_prob_1000, "b")
    plt.xlabel("Number of input memories")
    plt.ylabel("Probability of incorrectly flipping a bit")
    plt.legend(["100 neurons", "1000 neurons"])
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("q2 theoretical error probability N = 100, 1000")
    plt.close()

def q3():
    n_neurons = 100
    num_trials, num_repeats = 50, 100
    # num_trials, num_repeats = 3, 5
    n_memories_list = np.unique(np.logspace(0, 3, num_trials, dtype=np.int))
    num_trials = n_memories_list.shape[0]
    print(n_memories_list)
    error_prob_table = np.empty([num_trials, num_repeats])
    t_start = time()
    # Calculate error probability for each number of memories
    for i, n_memories in enumerate(n_memories_list):
        print("n_memories =", n_memories)
        # Repeat each experiment a few times
        for repeat in range(num_repeats):
            # Generate list of memories
            memories = gen_random_memories(n_neurons, n_memories)
            # Choose random memory as initial state
            initial_state = memories[np.random.choice(n_memories)]
            # Simulate Hopfield network
            output = simulate_hopfield_net(memories, initial_state, True)
            error_prob_table[i, repeat] = np.mean(output != initial_state)
    
    print("\nTime taken = {:.5}".format(time() - t_start))

    mean = error_prob_table.mean(axis=1)

    # Plot results
    n_memories = np.logspace(0, 3, 100)
    theoretical_error_probs = theoretical_error_prob(n_neurons, n_memories)
    
    plt.figure(figsize=[8, 6])
    handles = [
        plt.semilogx(
            n_memories_list, error_prob_table, "bo", alpha=0.03,
            markeredgewidth=0.0
        )[0],
        plt.semilogx(n_memories_list, mean, "b")[0],
        plt.semilogx(n_memories, theoretical_error_probs, "r--")[0],
    ]
    plt.xlabel("Number of input memories")
    plt.ylabel("Error probability")
    plt.legend(handles, [
        "Error fraction within an experiment",
        "Mean error fraction across experiments",
        "Theoretical error probability"
    ], bbox_to_anchor=(0.5, -0.1), loc="upper center")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("q3 simulated vs theoretical error probability")
    plt.close()


if __name__ == "__main__":
    # np.random.seed(0)
    q1()
    q2()
    q3()
