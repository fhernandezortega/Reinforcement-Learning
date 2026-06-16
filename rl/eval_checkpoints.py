"""
eval_checkpoints.py — Evaluacion de checkpoints para Fig. 2c
=============================================================
Referencia: PIPI2026 Fig. 2c

Para cada checkpoint guardado durante el entrenamiento:
  1. Carga el modelo QNetwork
  2. Evalua 1000 episodios en modo greedy (explore=False)
  3. Registra mean_steps y success_rate

La curva resultante (mean_steps vs episodio de entrenamiento)
reproduce la curva verde de la Fig. 2c del paper.

El paper reporta:
  - avg_steps RL evaluado: ~8.3
  - convergencia near-optimal: ~250 episodios
  - convergencia optima: ~550 episodios
"""

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import numpy as np
import json

from env.rlqls_env_cah import RLQLSEnvCaH
from rl.dqn import QNetwork


# =====================================================================
# Entorno de evaluacion
# =====================================================================
# Mismo entorno que el entrenamiento para comparacion justa

env = RLQLSEnvCaH(
    T=300.0,
    purity_threshold=0.99,
    ro=1.0,
    use_table_s2_only=False,
)

print(
    f"Entorno: {env.n_states} estados, "
    f"{env.n_actions} acciones",
    flush=True
)


# =====================================================================
# Funcion de evaluacion
# =====================================================================

def evaluate_model(model: QNetwork,
                   n_episodes: int = 1000,
                   seed: int = 0) -> dict:
    """
    Evalua el modelo en modo greedy (sin exploracion).

    Reproduce el proceso de la Fig. 2c: para cada modelo guardado
    en el entrenamiento, se corren 1000 episodios deterministas
    y se promedia el numero de pasos.

    Parameters
    ----------
    model      : QNetwork cargado desde checkpoint
    n_episodes : episodios de evaluacion (paper usa 1000)
    seed       : semilla para reproducibilidad

    Returns
    -------
    dict con mean_steps, std_steps, success_rate, step_counts
    """
    rng = np.random.default_rng(seed)
    model.eval()

    step_counts = []
    successes   = 0

    for ep in range(n_episodes):

        # Reset al estado termico de Boltzmann
        state, _ = env.reset(seed=int(rng.integers(1e6)))
        done  = False
        steps = 0

        while not done:

            # Seleccion greedy — sin exploracion (Fig. 2c)
            with torch.no_grad():
                s_t    = torch.tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0)
                action = int(model(s_t).argmax(dim=1).item())

            state, _, done, _, info = env.step(action)
            steps += 1

        step_counts.append(steps)

        if info["purity"] >= env.purity_threshold:
            successes += 1

    step_arr = np.array(step_counts)

    return {
        "mean_steps":   float(np.mean(step_arr)),
        "std_steps":    float(np.std(step_arr)),
        "success_rate": float(successes / n_episodes),
        "step_counts":  step_counts,
    }


# =====================================================================
# Cargar y evaluar cada checkpoint
# =====================================================================

checkpoint_dir = "checkpoints"

# Ordenar por numero de episodio
checkpoint_files = sorted(
    [f for f in os.listdir(checkpoint_dir)
     if f.startswith("model_ep") and f.endswith(".pt")],
    key=lambda x: int(x.replace("model_ep", "").replace(".pt", ""))
)

print(
    f"Encontrados {len(checkpoint_files)} checkpoints.\n",
    flush=True
)

results = {}

for fname in checkpoint_files:

    episode = int(
        fname.replace("model_ep", "").replace(".pt", "")
    )
    path = os.path.join(checkpoint_dir, fname)

    # Cargar QNetwork (misma arquitectura que el entrenamiento)
    model = QNetwork(
        n_states  = env.n_states,
        n_actions = env.n_actions,
        hidden_dims = [128, 128, 128],
    )
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"),
                   weights_only=False)
    )

    metrics = evaluate_model(model, n_episodes=1000, seed=42)

    results[str(episode)] = {
        "mean_steps":   metrics["mean_steps"],
        "std_steps":    metrics["std_steps"],
        "success_rate": metrics["success_rate"],
    }

    print(
        f"Ep {episode:5d} | "
        f"mean_steps={metrics['mean_steps']:5.2f} | "
        f"std={metrics['std_steps']:5.2f} | "
        f"success={metrics['success_rate']:.3f}",
        flush=True
    )

# =====================================================================
# Guardar resultados
# =====================================================================

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResultados guardados en eval_results.json", flush=True)

# =====================================================================
# Resumen final — comparacion con el paper
# =====================================================================

episodes_sorted = sorted(results.keys(), key=int)
steps_sorted    = [results[e]["mean_steps"] for e in episodes_sorted]

best_ep   = episodes_sorted[int(np.argmin(steps_sorted))]
best_mean = min(steps_sorted)
final_mean = steps_sorted[-1]

print(f"\nResumen:")
print(f"  Mejor modelo:    episodio {best_ep} → mean_steps={best_mean:.2f}")
print(f"  Modelo final:    episodio {episodes_sorted[-1]} → mean_steps={final_mean:.2f}")
print(f"  Paper reporta:   mean_steps ≈ 8.3 (RL) vs 9.7 (sweeping)")