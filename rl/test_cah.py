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
from collections import Counter

from env.rlqls_env_cah import RLQLSEnvCaH
from rl.dqn import DQN


# ==========================================================
# Entorno
# ==========================================================

env = RLQLSEnvCaH(T=300.0)

# Override max_steps para test
env.max_steps = 50

# Override reset para test con mezcla uniforme
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

print(
    f"States: {env.n_states}, "
    f"Actions: {env.n_actions}",
    flush=True
)


# ==========================================================
# Modelo
# ==========================================================

model = DQN(
    n_states=env.n_states,
    n_actions=env.n_actions,
)


# ==========================================================
# Cargar checkpoint
# ==========================================================

checkpoint = torch.load(
    "dqn_cah_model.pt",
    map_location=torch.device("cpu")
)

if isinstance(checkpoint, dict) and (
    "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint cargado "
        f"(episode {checkpoint['episode']})",
        flush=True
    )

else:

    model.load_state_dict(
        checkpoint
    )

    print(
        "Pesos antiguos cargados.",
        flush=True
    )


# ==========================================================
# Modo evaluación
# ==========================================================

model.eval()

print(
    "Model loaded.",
    flush=True
)


# ==========================================================
# Ejecutar episodios de prueba
# ==========================================================

n_test = 100

success = 0

step_counts = []

all_trajectories = []

print(
    "\nTesting agent...\n",
    flush=True
)

for episode in range(n_test):

    state, _ = env.reset()

    done = False

    steps = 0

    trajectory = []

    print(
        f"\n{'='*60}",
        flush=True
    )

    print(
        f"Episode {episode}",
        flush=True
    )

    print(
        f"{'='*60}",
        flush=True
    )

    print(
        "\nInitial state population:",
        flush=True
    )

    print(
        np.round(state, 4),
        flush=True
    )

    while not done:

        with torch.no_grad():

            state_t = torch.FloatTensor(
                state
            )

            q_values = model(state_t)

            action = torch.argmax(
                q_values
            ).item()

        if hasattr(env, "action_labels"):

            action_label = (
                env.action_labels[action]
            )

        else:

            action_label = (
                f"Action {action}"
            )

        print(
            f"\nStep {steps}",
            flush=True
        )

        print(
            "-" * 40,
            flush=True
        )

        print(
            f"Chosen action: "
            f"{action} | "
            f"{action_label}",
            flush=True
        )

        print(
            "\nQ-values:",
            flush=True
        )

        print(
            np.round(
                q_values.numpy(),
                4
            ),
            flush=True
        )

        print(
            "\nCurrent state population:",
            flush=True
        )

        print(
            np.round(state, 4),
            flush=True
        )

        next_state, reward, done, _, info = (
            env.step(action)
        )

        purity = info["purity"]

        trajectory.append(
            {
                "step": steps,
                "action": action,
                "action_label": action_label,
                "reward": reward,
                "purity": purity,
                "state_before": state.copy(),
                "state_after": next_state.copy(),
                "q_values": q_values.numpy().copy(),
            }
        )

        print(
            "\nReward:",
            reward,
            flush=True
        )

        print(
            f"Purity: {purity:.6f}",
            flush=True
        )

        print(
            "\nNext state population:",
            flush=True
        )

        print(
            np.round(next_state, 4),
            flush=True
        )

        state = next_state

        steps += 1

    if purity > env.purity_threshold:

        success += 1

        status = "SUCCESS"

    else:

        status = "TIMEOUT"

    step_counts.append(steps)

    all_trajectories.append(trajectory)

    print(
        f"\n{'-'*60}",
        flush=True
    )

    print(
        f"Episode {episode} finished",
        flush=True
    )

    print(
        f"Status       : {status}",
        flush=True
    )

    print(
        f"Final purity : {purity:.6f}",
        flush=True
    )

    print(
        f"Total steps  : {steps}",
        flush=True
    )

    print(
        f"{'-'*60}\n",
        flush=True
    )


# ==========================================================
# Estadísticas globales
# ==========================================================

print(
    f"\n{'='*60}",
    flush=True
)

print(
    "FINAL TEST RESULTS",
    flush=True
)

print(
    f"{'='*60}",
    flush=True
)

print(
    f"\nSuccess rate: "
    f"{success}/{n_test} = "
    f"{100 * success / n_test:.1f}%",
    flush=True
)

print(
    f"Mean steps: "
    f"{np.mean(step_counts):.3f}",
    flush=True
)

print(
    f"Min steps: "
    f"{np.min(step_counts)}",
    flush=True
)

print(
    f"Max steps: "
    f"{np.max(step_counts)}",
    flush=True
)

print(
    f"{'='*60}\n",
    flush=True
)


# ==========================================================
# Distribucion de estados terminales (1000 episodios)
# ==========================================================

print(
    "\nCalculando distribucion de estados terminales "
    "(1000 episodios)...",
    flush=True
)

terminal_states = []

for episode in range(1000):

    state, _ = env.reset()

    done = False

    while not done:

        with torch.no_grad():

            q_values = model(
                torch.FloatTensor(state)
            )

            action = torch.argmax(
                q_values
            ).item()

        state, reward, done, _, info = (
            env.step(action)
        )

    if info["purity"] > env.purity_threshold:

        terminal_states.append(
            int(np.argmax(state))
        )

labels = env.ham.get_labels()

print(
    "\nDistribucion de estados terminales:",
    flush=True
)

for idx, count in sorted(
    Counter(terminal_states).items()
):

    print(
        f"  Estado {idx:2d} {labels[idx]}: "
        f"{count} veces "
        f"({100*count/len(terminal_states):.1f}%)",
        flush=True
    )

print(
    f"\nTotal exitosos: "
    f"{len(terminal_states)}/1000",
    flush=True
)