import json
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Load history
# =====================================

with open("training_history.json", "r") as f:

    history = json.load(f)

steps = np.array(history["steps"])

episodes = np.arange(1, len(steps) + 1)

# =====================================
# Moving average (100 episodes)
# =====================================

window = 100

avg_steps = np.convolve(
    steps,
    np.ones(window) / window,
    mode="valid"
)

avg_episodes = np.arange(
    window,
    len(steps) + 1
)

# =====================================
# Plot
# =====================================

plt.figure(figsize=(4,4))

# orange individual episodes
plt.plot(
    episodes,
    steps,
    color="orange",
    linewidth=1,
    alpha=0.7,
    label="RL individual"
)

# blue moving average
plt.plot(
    avg_episodes,
    avg_steps,
    color="blue",
    linewidth=2.5,
    label="RL averaged"
)

# sweeping protocol baseline
plt.axhline(
    y=9,
    color="magenta",
    linewidth=2,
    label="sweeping protocol"
)

plt.xlabel("# training episodes")
plt.ylabel("# steps per episode")

plt.xlim(0, len(steps))
plt.ylim(0, 30)

plt.legend()

plt.tight_layout()

plt.show()