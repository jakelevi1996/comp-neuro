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
    window_counts = cum_spikes[n_window_steps:] - cum_spikes[:-n_window_steps]
    
    fano = window_counts.var() / window_counts.mean()
    return fano

def connection_matrix(N, K):
    C = np.zeros([N, N])
    for row in C: row[np.random.choice(N, K, replace=False)] = 1.0
    
    return C

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

def q5a():
    # Initialise synaptic weights for different populations
    J_EE = 1.0
    J_IE = 1.0
    J_EI = -2.0
    J_II = -1.8
    J_EX = 1.0
    J_IX = 0.8

    A = np.array([[J_EE, -J_EI], [J_IE, -J_II]])
    b = np.array([[-J_EX * r_X], [-J_IX * r_X]])
    print(A, b)
    r_E, r_I = np.linalg.solve(A, b).reshape(-1)
    print(r_E, r_I)
    

def q5b(r_X=10.0, N=1000, K=100, plot=True, t_end=2.0, filename=None):
    # Generate external spike trains
    t, spikes_X = gen_spike_trains(num_neurons=N, firing_rate=r_X, t_end=t_end)

    # Initialise excitory and inhibitory voltage signals and spike trains
    V_E = np.zeros([N, t.size])
    V_I = np.zeros([N, t.size])
    spikes_E = np.zeros([N, t.size])
    spikes_I = np.zeros([N, t.size])

    # Initialise synaptic weights for different populations
    J_EE = 1.0
    J_IE = 1.0
    J_EI = -2.0
    J_II = -1.8
    J_EX = 1.0
    J_IX = 0.8

    # Initialise connection matrices
    C_EE = connection_matrix(N, K)
    C_IE = connection_matrix(N, K)
    C_EI = connection_matrix(N, K)
    C_II = connection_matrix(N, K)
    C_EX = connection_matrix(N, K)
    C_IX = connection_matrix(N, K)
    
    for k in range(1, t.size):
        if k % 100 == 0: print("t = {:.4}".format(t[k]), end=" ", flush=True)
        # Update excitory membrane potentials
        V_E[:, k] = (1.0 - dt / tau) * V_E[:, k-1]
        V_E[:, k] += dt * J_EX * C_EX.dot(spikes_X[:, k-1]) / np.sqrt(K)
        V_E[:, k] += dt * J_EE * C_EE.dot(spikes_E[:, k-1]) / np.sqrt(K)
        V_E[:, k] += dt * J_EI * C_EI.dot(spikes_I[:, k-1]) / np.sqrt(K)
        # Update inhibitory membrane potentials
        V_I[:, k] = (1.0 - dt / tau) * V_I[:, k-1]
        V_I[:, k] += dt * J_IX * C_IX.dot(spikes_X[:, k-1]) / np.sqrt(K)
        V_I[:, k] += dt * J_IE * C_IE.dot(spikes_E[:, k-1]) / np.sqrt(K)
        V_I[:, k] += dt * J_II * C_II.dot(spikes_I[:, k-1]) / np.sqrt(K)
        
        # Update spike trains
        spikes_E[V_E[:, k] > V_th, k] = 1.0 / dt
        spikes_I[V_I[:, k] > V_th, k] = 1.0 / dt
        # Reset spiking neurons
        V_E[V_E[:, k] > V_th, k] = 0.0
        V_I[V_I[:, k] > V_th, k] = 0.0
    
    r_E = np.sum(spikes_E > 0, axis=1).mean() / t_end
    r_I = np.sum(spikes_I > 0, axis=1).mean() / t_end

    print("\n\n", r_E, r_I)
    if plot:
        if filename is None: filename = "q5b population spikes"
        plotting.plot_populations_spikes(
            t, spikes_X, spikes_E, spikes_I, filename
        )
    return r_E, r_I

def q5c():
    r_X_list = [5, 10, 15, 20]
    r_E_list, r_I_list = [], []
    for r_X in r_X_list:
        r_E, r_I = q5b(r_X, plot=False)
        r_E_list.append(r_E)
        r_I_list.append(r_I)
    
    plotting.plot_firing_rates(
        r_X_list, r_E_list, r_I_list, "q5c internal firing rates"
    )
    
def q5d():
    q5b(N=100, filename="q5d fewer neurons")


if __name__ == "__main__":
    # q1()
    # q2()
    # q3a()
    # q3a(w=1.0)
    # q3c()
    # q3d()
    # q3e()
    # q4()
    
    # for _ in range(5): q4()

    q5a()
    q5b()
    q5c()
    q5d()
