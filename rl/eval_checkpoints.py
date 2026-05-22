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
import json

from env.rlqls_env_cah import RLQLSEnvCaH
from rl.dqn import DQN


# =========================
# Environment
# =========================

env = RLQLSEnvCaH(
    T=300.0,
    purity_threshold=0.99
)

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
env.max_steps = 50

# =========================
# Evaluation function
# =========================

def evaluate_model(model, n_episodes=1000):

    step_counts = []
    successes   = 0

    for _ in range(n_episodes):

        state, _ = env.reset()
        done      = False
        steps     = 0

        while not done:

            with torch.no_grad():

                q_values = model(
                    torch.FloatTensor(state)
                )

                action = torch.argmax(
                    q_values
                ).item()

            state, _, done, _, info = (
                env.step(action)
            )

            steps += 1

        step_counts.append(steps)

        if info["purity"] > env.purity_threshold:
            successes += 1

    return {
        "mean_steps":   float(np.mean(step_counts)),
        "success_rate": float(successes / n_episodes),
    }


# =========================
# Load and evaluate each
# checkpoint
# =========================

checkpoint_dir = "checkpoints"

results = {}

checkpoint_files = sorted([
    f for f in os.listdir(checkpoint_dir)
    if f.endswith(".pt")
],
key=lambda x: int(
    x.replace("model_ep", "").replace(".pt", "")
))

print(
    f"Encontrados {len(checkpoint_files)} "
    f"checkpoints.",
    flush=True
)

for fname in checkpoint_files:

    episode = int(
        fname.replace(
            "model_ep", ""
        ).replace(".pt", "")
    )

    path = os.path.join(checkpoint_dir, fname)

    model = DQN(
        n_states=env.n_states,
        n_actions=env.n_actions,
    )

    model.load_state_dict(
        torch.load(
            path,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    metrics = evaluate_model(model)

    results[episode] = metrics

    print(
        f"Episode {episode:4d} | "
        f"mean_steps={metrics['mean_steps']:.2f} | "
        f"success_rate={metrics['success_rate']:.3f}",
        flush=True
    )

# =========================
# Save results
# =========================

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(
    "\nResultados guardados en eval_results.json",
    flush=True
)