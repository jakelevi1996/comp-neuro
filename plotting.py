import matplotlib.pyplot as plt
import numpy as np

# Plot reward and stimulus
def plot_stimulus_and_reward(T, s, r, filename="Input signals"):
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
    T, V, TD, learning_error, N_trials=201, every_nth_trial=10, a=0.5,
    title="Output signals for tapped delay-line TD-learning",
    filename="tapdl output signals"
):
    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(8, 6)
    trials_to_plot = range(0, N_trials, every_nth_trial)
    colours = iter(plt.cm.get_cmap("winter")(
        np.linspace(1, 0, len(trials_to_plot))
    ))
    handles, labels = [], []

    for trial in trials_to_plot:
        colour = next(colours)
        labels.append("{} trials".format(trial))
        handles.append(axes[0].plot(T, V[trial], c=colour, alpha=a)[0])
        axes[1].plot(T, TD[trial], c=colour, alpha=a)
        axes[2].plot(T, learning_error[trial], c=colour, alpha=a)
    for a in axes: a.grid(True)
    fig.legend(handles, labels, "center right")
    axes[0].set_title(title, fontsize=15)
    axes[0].set_ylabel("State value")
    axes[1].set_ylabel("Temporal difference")
    axes[2].set_ylabel("Learning error")
    axes[2].set_xlabel("Time (s)")
    plt.savefig(filename)
    plt.close()