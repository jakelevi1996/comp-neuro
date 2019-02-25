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


if __name__ == "__main__":
    # q1()
    q2()
