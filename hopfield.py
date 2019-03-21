import numpy as np
import matplotlib.pyplot as plt
import plotting
from scipy.stats import norm

def theoretical_error_prob(n_neurons, n_memories):
    return norm.cdf(-np.sqrt((n_neurons - 1) / (2 * n_memories - 1)))



def q1(n_neurons=100):
    n_memories = np.logspace(0, 3, 50)
    error_prob = theoretical_error_prob(n_neurons, n_memories)
    
    # TODO:  move plotting code to `plotting` module
    plt.figure(figsize=[8, 6])
    plt.semilogx(n_memories, error_prob, "bo--")
    plt.xlabel("Number of input memories")
    plt.ylabel("Probability of flipping a bit")
    # plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("q1 theoretical error probability N = 100")
    plt.close()


def q2():
    # TODO: rewrite this with better code reuse
    n_memories = np.logspace(0, 3, 50)
    error_prob_100 = theoretical_error_prob(100, n_memories)
    error_prob_1000 = theoretical_error_prob(1000, n_memories)
    
    plt.figure(figsize=[8, 6])
    plt.semilogx(n_memories, error_prob_100, "bo--")
    plt.semilogx(n_memories, error_prob_1000, "ro--")
    plt.xlabel("Number of input memories")
    plt.ylabel("Probability of flipping a bit")
    # plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("q2 theoretical error probability N = 100, 1000")
    plt.close()

if __name__ == "__main__":
    # q1()
    q2()
