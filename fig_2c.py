import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import torch
import numpy as np
import matplotlib.pyplot as plt

from env.rlqls_env_cah import RLQLSEnvCaH
from rl.dqn import DQN

# =====================================
# Environment
# =====================================

env = RLQLSEnvCaH(
    T=300.0,
    purity_threshold=0.99
)

env.max_steps = 50

# Fixed reset for evaluation
def fixed_reset(seed=None, options=None):
    env.state = np.zeros(
        env.n_states,
        dtype=np.float32
    )
    for i in range(6):
        env.state[i] = 1.0 / 6.0
    env.steps = 0
    return env.state, {}

env.reset = fixed_reset

# =====================================
# Testing parameters
# =====================================

test_episodes = 1000

checkpoint_steps = list(range(50, 1001, 50))

avg_test_steps   = []
success_rates    = []

# =====================================
# Test each checkpoint
# =====================================

for ckpt in checkpoint_steps:

    path = f"checkpoints/model_ep{ckpt}.pt"

    if not os.path.exists(path):
        print(
            f"Checkpoint {ckpt} no encontrado, "
            f"saltando...",
            flush=True
        )
        continue

    print(
        f"Testing checkpoint {ckpt}...",
        flush=True
    )

    model = DQN(
        n_states=env.n_states,
        n_actions=env.n_actions
    )

    model.load_state_dict(
        torch.load(
            path,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    steps_list = []
    successes  = 0

    for ep in range(test_episodes):

        state, _ = env.reset()

        done = False

        step_count = 0

        while not done:

            with torch.no_grad():

                state_t = torch.FloatTensor(state)

                q_values = model(state_t)

                action = torch.argmax(
                    q_values
                ).item()

            next_state, reward, done, _, info = (
                env.step(action)
            )

            state = next_state

            step_count += 1

        steps_list.append(step_count)

        if info["purity"] > env.purity_threshold:
            successes += 1

    avg_steps = np.mean(steps_list)
    success   = successes / test_episodes

    avg_test_steps.append(avg_steps)
    success_rates.append(success)

    print(
        f"  mean_steps={avg_steps:.2f} | "
        f"  success={100*success:.1f}%",
        flush=True
    )

# =====================================
# Plot Fig. 2(c)
# =====================================

valid_checkpoints = [
    c for c in checkpoint_steps
    if os.path.exists(
        f"checkpoints/model_ep{c}.pt"
    )
]

fig, ax1 = plt.subplots(figsize=(5, 4))

ax1.plot(
    valid_checkpoints,
    avg_test_steps,
    color="green",
    linewidth=2.5,
    label="RL testing"
)

ax1.axhline(
    y=9.7,
    color="purple",
    linewidth=1.5,
    linestyle="--",
    label="sweeping protocol"
)

ax1.set_xlabel("# training episodes")
ax1.set_ylabel("# steps per episode")
ax1.set_xlim(0, 1000)
ax1.set_ylim(0, 30)
ax1.legend()

plt.tight_layout()
plt.savefig("fig2c.png", dpi=300)
plt.show()

print("Fig. 2c guardada.", flush=True)