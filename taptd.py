import numpy as np
import matplotlib.pyplot as plt

# Define time
t = np.arange(0, 25, 0.5)
# Define stimulus (IE state)
s = np.zeros(t.shape)
s[t == 10] = 1
# Define reward
r = 0.5 * np.exp(-0.5 * np.square(t - 20))

# Plot reward and stimulus
plt.figure(figsize=[16, 9])
plt.plot(t, s, 'r', t, r, 'g')
plt.legend(["Stimulus", "Reward"])
plt.xlabel("Time (s)")
plt.ylabel("Signal amplitude")
plt.title("Input signals to tapped delay line TD")
plt.grid(True)
plt.savefig("taptd input signals")