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
    

if __name__ == "__main__":
    q1()