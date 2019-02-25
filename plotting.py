import matplotlib.pyplot as plt
import numpy as np

# Plot reward and stimulus
def plot_stimulus_and_reward(T, s, r, filename):
    plt.figure(figsize=[8, 6])
    plt.plot(T, s, 'r', T, r, 'b', alpha=0.5)
    plt.legend(["Stimulus", "Reward"])
    plt.xlabel("Time (s)")
    plt.ylabel("Signal amplitude")
    plt.title("Input signals to tapped delay line TD")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Plot state value, temporal difference, and learning error
def plot_value_td_learning_error(
    T, V, TD, learning_error, filename, title,
    N_trials=201, every_nth_trial=10, a=0.5
):
    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(8, 6)
    trials_to_plot = range(0, N_trials, every_nth_trial)
    cmap = plt.cm.get_cmap("winter")
    colours = iter(cmap(np.linspace(1, 0, len(trials_to_plot))))
    handles, labels = [], []

    for trial in trials_to_plot:
        colour = next(colours)
        labels.append("{} trials".format(trial))
        handles.extend(axes[0].plot(T, V[trial], c=colour, alpha=a))
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

def plot_partial_reinforcements(
    T, V, TD, learning_error, rewarded_trials, filename, title,
    last_N_trials=100, a=0.5
):
    # Truncate arrays to last few trials
    V = V[-last_N_trials:]
    TD = TD[-last_N_trials:]
    learning_error = learning_error[-last_N_trials:]
    rewarded_trials = rewarded_trials[-last_N_trials:]
    # Create figure and subplot axes
    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(8, 6)
    # Plot averaged signals
    handles = axes[0].plot(
        T, V[rewarded_trials == 1].mean(axis=0), "g",
        T, V[rewarded_trials == 0].mean(axis=0), "r",
        T, V.mean(axis=0), "b", alpha=a
    )
    axes[1].plot(
        T, TD[rewarded_trials == 1].mean(axis=0), "g",
        T, TD[rewarded_trials == 0].mean(axis=0), "r",
        T, TD.mean(axis=0), "b", alpha=a
    )
    axes[2].plot(
        T, learning_error[rewarded_trials == 1].mean(axis=0), "g",
        T, learning_error[rewarded_trials == 0].mean(axis=0), "r",
        T, learning_error.mean(axis=0), "b", alpha=a
    )
    # Format the figure and save
    for a in axes: a.grid(True)
    labels = ["Rewarded trials", "Unrewarded trials", "All trials"]
    fig.legend(handles, labels, "center right")
    axes[0].set_title(title, fontsize=15)
    axes[0].set_ylabel("State value")
    axes[1].set_ylabel("Temporal difference")
    axes[2].set_ylabel("Learning error")
    axes[2].set_xlabel("Time (s)")
    plt.savefig(filename)
    plt.close()

def plot_dopamine_vs_learning_error(
    T, dopamine_activity, learning_error, filename,
    last_N_trials=100, a=0.5
):
    # Truncate arrays to last few trials
    dopamine_activity = dopamine_activity[-last_N_trials:].mean(axis=0)
    learning_error = learning_error[-last_N_trials:].mean(axis=0)
    # Create figure and plot
    plt.figure()
    plt.plot(T, learning_error, "r", T, dopamine_activity, "b", alpha=a)
    # Format figure
    plt.grid(True)
    plt.title("Time course of dopamine signal")
    plt.ylabel("Signal amplitude")
    plt.xlabel("Time (s)")
    plt.legend(["Learning error", "Dopamine signal"])
    # Save and close
    plt.savefig(filename)
    plt.close()

def plot_probabilistic_dopamine(T, dopamine_signals, p_list, filename, a=0.5):
    cmap = plt.cm.get_cmap("winter")
    colours = iter(cmap(np.linspace(1, 0, len(p_list))))
    handles, labels = [], []
    plt.figure()
    for d, p in zip(dopamine_signals, p_list):
        handles.extend(plt.plot(T, d, c=next(colours), alpha=a))
        labels.append("p = {:.2}".format(p))
    plt.grid(True)
    plt.legend(handles, labels)
    plt.savefig(filename)
    plt.close()
