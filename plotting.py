import matplotlib.pyplot as plt

# Plot reward and stimulus
def plot_stimulus_and_reward(T, s, r, filename="taptd input signals"):
    plt.figure(figsize=[8, 6])
    plt.plot(T, s, 'rx-', T, r, 'bx-', alpha=0.5)
    plt.legend(["Stimulus", "Reward"])
    plt.xlabel("Time (s)")
    plt.ylabel("Signal amplitude")
    plt.title("Input signals to tapped delay line TD")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Plot state value, temporal difference, and learning error
def plot_value_td_learning_error(
    T, V, TD, learning_error, N_trials=201, every_nth_trial=10,
    filename="taptd output signals"
):
    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(16, 9)
    for trial in range(0, N_trials, every_nth_trial):
        axes[0].plot(T, V[trial])
        axes[1].plot(T, TD[trial])
        axes[2].plot(T, learning_error[trial])
    for a in axes:
        a.grid(True)
        # a.set_ylim(-2,6)
    plt.savefig(filename)
    plt.close()