"""
train_cah.py — Loop de entrenamiento RL-QLS para CaH+ (J<=2)
==============================================================
Referencia: PIPI2026 Fig. 2b, Sec. SD

Configuracion actual:
  - 48 pulsos auto-enumerados (reglas E1)
  - R = -1 + penalizacion estancamiento (Sec. SD)
  - qMDP update (Ec. S18): incorpora p0, p1, s'_k0, s'_k1
    en el replay buffer para el update de TD
"""

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import numpy as np
import json
import torch

from env.rlqls_env_cah import RLQLSEnvCaH
from rl.dqn import DQNAgent


# =====================================================================
# Reproducibilidad
# =====================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# =====================================================================
# Entorno
# =====================================================================

env = RLQLSEnvCaH(
    T=300.0,
    purity_threshold=0.99,
    ro=1.0,                    # peso penalizacion estancamiento
    use_table_s2_only=False,   # 48 pulsos auto-enumerados
)

print(
    f"States: {env.n_states}, "
    f"Actions: {env.n_actions}",
    flush=True
)

os.makedirs("checkpoints", exist_ok=True)


# =====================================================================
# Hiperparametros (Sec. SD, Tabla S1)
# =====================================================================

episodes = 1500   # mas episodios para convergencia con 48 acciones


# =====================================================================
# Agente DQN con qMDP
# =====================================================================

agent = DQNAgent(
    n_states        = env.n_states,
    n_actions       = env.n_actions,
    hidden_dims     = [128, 128, 128],  # Sec. SD
    buffer_capacity = 50_000,           # mayor buffer para qMDP
    batch_size      = 64,               # batch mas grande para qMDP
    min_buffer      = 64,
    lr              = 5e-4,             # Tabla S1
    gamma           = 1.0,              # sin descuento
    eps_start       = 1.0,
    eps_end         = 0.005,            # Sec. SD
    N_training      = episodes,
    tau_update      = 0.001,            # Tabla S1
    loss_type       = 'smooth_l1',
    use_qmdp        = True,             # qMDP (Ec. S18)
)

print(f"\n{agent}", flush=True)


# =====================================================================
# Estadisticas
# =====================================================================

reward_history = []
step_history   = []
purity_history = []


# =====================================================================
# Loop de entrenamiento con qMDP
# =====================================================================

for episode in range(episodes):

    state, _ = env.reset()
    done         = False
    total_reward = 0.0
    step_count   = 0

    while not done:

        # Seleccion epsilon-greedy
        action = agent.select_action(state, explore=True)

        # ── Precomputar ramas qMDP ANTES del step ────────────────────
        # Necesario para almacenar p0, p1, s'_k0, s'_k1 en el buffer
        # segun Ec. S18 sin re-ejecutar el simulador en el update
        A0, A1 = env.get_transition_matrices(action)
        v0 = A0 @ state
        v1 = A1 @ state
        p0 = float(v0.sum())
        p1 = float(v1.sum())
        total = p0 + p1 + 1e-12
        p0 /= total
        p1 /= total

        # Normalizar estados post-medicion
        s_k0 = v0 / (v0.sum() + 1e-12) if v0.sum() > 1e-10 \
               else np.ones(env.n_states) / env.n_states
        s_k1 = v1 / (v1.sum() + 1e-12) if v1.sum() > 1e-10 \
               else np.ones(env.n_states) / env.n_states

        # ── Paso del entorno ─────────────────────────────────────────
        next_state, reward, done, _, info = env.step(action)

        total_reward += reward
        step_count   += 1

        # ── Almacenar con datos qMDP ──────────────────────────────────
        # Ec. S18: el buffer guarda ambas ramas para el TD update
        agent.store(
            state, action, reward, next_state, done,
            p0=p0, p1=p1,
            next_s_k0=s_k0.astype(np.float32),
            next_s_k1=s_k1.astype(np.float32),
        )

        state = next_state

        # ── qMDP update (Ec. S18) ─────────────────────────────────────
        agent.update()

    # ── Fin de episodio ───────────────────────────────────────────────
    agent.decay_epsilon()

    # ── Logging ───────────────────────────────────────────────────────
    reward_history.append(total_reward)
    step_history.append(step_count)
    purity_history.append(info["purity"])

    avg_steps  = np.mean(step_history[-100:])
    avg_purity = np.mean(purity_history[-100:])

    print(
        f"Episode {episode:4d} | "
        f"steps={step_count:3d} | "
        f"reward={total_reward:8.2f} | "
        f"purity={info['purity']:.3f} | "
        f"avg_steps={avg_steps:5.2f} | "
        f"avg_purity={avg_purity:.3f} | "
        f"epsilon={agent.eps:.3f}",
        flush=True
    )

    # ── Checkpoint cada 50 episodios (para Fig. 2c) ───────────────────
    if (episode + 1) % 50 == 0:
        path = f"checkpoints/model_ep{episode+1}.pt"
        agent.save(path)
        print(f"Checkpoint guardado: {path}", flush=True)


# =====================================================================
# Guardar modelo final e historial
# =====================================================================

agent.save("dqn_cah_model.pt")

history = {
    "steps":   step_history,
    "rewards": reward_history,
    "purity":  purity_history,
}

with open("training_history.json", "w") as f:
    json.dump(history, f)

print("Model saved.", flush=True)
print("Historia guardada.", flush=True)