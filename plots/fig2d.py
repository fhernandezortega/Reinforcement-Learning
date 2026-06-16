import sys
import os

# ==========================================
# Agregar raíz del proyecto al PATH
# ==========================================

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


# ==========================================
# Parámetros
# ==========================================

MAX_DEPTH = 10
PURITY_THRESHOLD = 0.99


# ==========================================
# Environment
# ==========================================

env = RLQLSEnvCaH(
    T=300.0,
    purity_threshold=PURITY_THRESHOLD
)

print(
    f"Estados: {env.n_states} | "
    f"Acciones: {env.n_actions}",
    flush=True
)


# ==========================================
# Modelo entrenado
# ==========================================

model = DQN(
    n_states=env.n_states,
    n_actions=env.n_actions
)

model.load_state_dict(
    torch.load("dqn_cah_model.pt")
)

model.eval()

print("Modelo cargado.", flush=True)


# ==========================================
# Expansión recursiva del árbol
# ==========================================

def expand_tree(
    state,
    depth,
    branch_probability,
    history
):

    # ======================================
    # Limitar profundidad
    # ======================================

    if depth >= MAX_DEPTH:

        return None

    purity = np.max(state)

    # ======================================
    # Estado terminal
    # ======================================

    if purity >= PURITY_THRESHOLD:

        return {

            "type": "terminal",

            "depth": depth,

            "probability": branch_probability,

            "terminal_state": int(
                np.argmax(state)
            ),

            "purity": float(purity),

            "history": history
        }

    # ======================================
    # Inferencia DQN
    # ======================================

    state_t = torch.FloatTensor(state)

    with torch.no_grad():

        q_values = model(state_t)

        action = torch.argmax(
            q_values
        ).item()

    # ======================================
    # Obtener información del pulso
    # CORREGIDO
    # ======================================

    pulse = env.ACTIONS[action]

    # ======================================
    # Matrices de transición
    # ======================================

    A0, A1 = env.get_transition_matrices(action)

    # ======================================
    # Evolución probabilística
    # ======================================

    s0 = A0 @ state
    s1 = A1 @ state

    p0 = float(np.sum(s0))
    p1 = float(np.sum(s1))

    # ======================================
    # Normalizar estados
    # ======================================

    if p0 > 1e-12:

        s0 = s0 / p0

    if p1 > 1e-12:

        s1 = s1 / p1

    # ======================================
    # Nodo actual
    # ======================================

    node_info = {

        "type": "node",

        "depth": depth,

        "probability": branch_probability,

        "action": action + 1,

        "pulse_label": pulse["label"],

        "p(k=0)": p0,

        "p(k=1)": p1,

        "purity": float(purity),

        "history": history,

        "children": {}
    }

    # ======================================
    # Rama k=0
    # ======================================

    if p0 > 1e-8:

        next_history_0 = history + [

            {
                "action": action + 1,

                "measurement": "k=0",

                "probability": p0
            }
        ]

        child_0 = expand_tree(

            s0,

            depth + 1,

            branch_probability * p0,

            next_history_0
        )

        if child_0:

            node_info["children"]["k=0"] = child_0

    # ======================================
    # Rama k=1
    # ======================================

    if p1 > 1e-8:

        next_history_1 = history + [

            {
                "action": action + 1,

                "measurement": "k=1",

                "probability": p1
            }
        ]

        child_1 = expand_tree(

            s1,

            depth + 1,

            branch_probability * p1,

            next_history_1
        )

        if child_1:

            node_info["children"]["k=1"] = child_1

    return node_info


# ==========================================
# Construcción del árbol
# ==========================================

initial_state, _ = env.reset()

print(
    "Construyendo árbol...",
    flush=True
)

tree_root = expand_tree(

    state=initial_state,

    depth=0,

    branch_probability=1.0,

    history=[]
)


# ==========================================
# Guardar JSON
# ==========================================

with open(
    "tree_data_structured.json",
    "w"
) as f:

    json.dump(
        tree_root,
        f,
        indent=2
    )

print(
    "Árbol guardado correctamente.",
    flush=True
)