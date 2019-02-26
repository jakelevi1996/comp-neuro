import numpy as np
import plotting

# Define constants
V_th = 1.0
dt = 1e-4
tau = 0.02
N = 1000
K = 100
r_X = 10

def gen_spike_trains(t_end=2.0, dt=dt, firing_rate=r_X, num_neurons=K):
    t = np.arange(0, t_end, dt)
    p = firing_rate * dt
    spikes = np.random.binomial(n=1, p=p, size=[num_neurons, t.size]) / dt
    return t, spikes

def estimate_fano(t, spikes, window_time=0.1):
    n_window_steps = np.sum(t < window_time)
    cum_spikes = np.cumsum(spikes > 0)
    counts = cum_spikes[n_window_steps:] - cum_spikes[:-n_window_steps]
    
    fano = counts.var() / counts.mean()
    return fano

def simulate_network_dynamics(): pass

def q1():
    t, spikes = gen_spike_trains()
    spike_coords = np.argwhere(spikes > 0)
    t_spikes = t[spike_coords[:, 1]]
    k_spikes = spike_coords[:, 0]
    plotting.plot_spike_raster_plot(t_spikes, k_spikes, "q1 spike raster plot")
    
    mean_sum = (spikes > 0).sum(axis=1).mean()
    print(mean_sum)

def q2():
    w = 0.9
    t, input_spikes = gen_spike_trains(num_neurons=1)
    input_spikes = input_spikes.ravel()

    V = np.zeros(t.size)
    V_above_th = np.zeros(t.size)
    output_spikes = np.zeros(t.size)
    
    for k in range(1, t.size):
        V[k] = (1.0 - dt / tau) * V[k-1] + dt * w * input_spikes[k-1]
        V_above_th[k] = V[k]
        if V[k] > V_th:
            V[k] = 0.0
            output_spikes[k] = 1.0 / dt
    
    plotting.plot_single_lif_neuron(
        t, input_spikes, V_above_th, V_th, output_spikes, "q2 single LIF neuron"
    )

def q3a(w=0.9):
    t, input_spikes = gen_spike_trains()
    summed_inputs = input_spikes.sum(axis=0)

    V = np.zeros(t.size)
    
    for k in range(1, t.size):
        V[k] = (1.0 - dt / tau) * V[k-1] + dt * w * summed_inputs[k-1] / K

    filename = "q3a single neuron many inputs no reset w={}.png".format(w)
    plotting.plot_single_lif_noreset(t, input_spikes, V, filename)

def q3c():
    w = 1.0
    K_list = np.logspace(0, 3, 10, dtype=np.int32)
    means, vars = [], []

    for K in K_list:
        print(K)
        t, input_spikes = gen_spike_trains(num_neurons=K, t_end=15.0)
        summed_inputs = input_spikes.sum(axis=0)

        V = np.zeros(t.size)
        
        for k in range(1, t.size):
            V[k] = (1.0 - dt / tau) * V[k-1] + dt * w * summed_inputs[k-1] / K
        
        means.append(V[t > 0.1].mean())
        vars.append(V[t > 0.1].var())
    
    t_means = tau * w * r_X * np.ones(K_list.shape)
    t_vars = w * w * r_X * tau / (2 * K_list)

    plotting.plot_stationary_statistics(
        K_list, means, vars, t_means, t_vars,
        "q3c single neuron stationary statistics"
    )

def q3d():
    w = V_th/ (tau * r_X)
    print(w)
    t, input_spikes = gen_spike_trains(num_neurons=K)
    summed_inputs = input_spikes.sum(axis=0)

    V = np.zeros(t.size)
    
    for k in range(1, t.size):
        V[k] = (1.0 - dt / tau) * V[k-1] + dt * w * summed_inputs[k-1] / K

    filename = "q3d single neuron many inputs no reset w={}.png".format(w)
    plotting.plot_single_lif_noreset(t, input_spikes, V, filename)

def q3e():
    w = 4.25
    t_end = 50.0
    t, input_spikes = gen_spike_trains(t_end=t_end)
    summed_inputs = input_spikes.sum(axis=0)

    V = np.zeros(t.size)
    V_above_th = np.zeros(t.size)
    output_spikes = np.zeros(t.size)
    
    for k in range(1, t.size):
        V[k] = (1.0 - dt / tau) * V[k-1] + dt * w * summed_inputs[k-1] / K
        V_above_th[k] = V[k]
        if V[k] > V_th:
            V[k] = 0.0
            output_spikes[k] = 1.0 / dt
    
    print("Output firing rate =", np.sum(output_spikes > 0) / t_end)
    fano = estimate_fano(t, output_spikes)
    print("Fano factor ~ {:.5}".format(fano))

    plotting.plot_neuron_spikes_many_inputs(
        t, input_spikes, V_above_th, V_th, output_spikes,
        "q3e single LIF neuron many inputs with reset"
    )

def q4():
    w = 1.5
    t_end = 50.0
    t, input_e_spikes = gen_spike_trains(t_end=t_end)
    summed_e_inputs = input_e_spikes.sum(axis=0)
    _, input_i_spikes = gen_spike_trains(t_end=t_end)
    summed_i_inputs = input_i_spikes.sum(axis=0)

    V = np.zeros(t.size)
    V_above_th = np.zeros(t.size)
    output_spikes = np.zeros(t.size)
    
    for k in range(1, t.size):
        V[k] = (1.0 - dt / tau) * V[k-1]
        V[k] += dt * w * summed_e_inputs[k-1] / np.sqrt(K)
        V[k] -= dt * w * summed_i_inputs[k-1] / np.sqrt(K)
        V_above_th[k] = V[k]
        if V[k] > V_th:
            V[k] = 0.0
            output_spikes[k] = 1.0 / dt
    
    print("Output firing rate =", np.sum(output_spikes > 0) / t_end)
    fano = estimate_fano(t, output_spikes)
    print("Fano factor ~ {:.5}".format(fano))

if __name__ == "__main__":
    # q1()
    # q2()
    # q3a()
    # q3a(w=1.0)
    # q3c()
    # q3d()
    # q3e()
    # q4()
    
    for _ in range(5): q4()
