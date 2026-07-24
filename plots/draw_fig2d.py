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
from rl.dqn import QNetwork          # <-- era 'DQN' (no existe)


# ==========================================
# Rutas robustas
# ==========================================

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

def find_file(name):
    for d in (HERE, ROOT, os.getcwd()):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, name)

MODEL_PATH = find_file("dqn_cah_model.pt")
JSON_PATH  = os.path.join(HERE, "tree_data_structured.json")


# ==========================================
# Parametros
# ==========================================

MAX_DEPTH        = 12
PURITY_THRESHOLD = 0.99


# ==========================================
# Environment
# ==========================================

env = RLQLSEnvCaH(T=300.0, purity_threshold=PURITY_THRESHOLD)
print(f"Estados: {env.n_states} | Acciones: {env.n_actions}", flush=True)


# ==========================================
# Modelo entrenado
# ==========================================

ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model = QNetwork(n_states=env.n_states, n_actions=env.n_actions)
if isinstance(ckpt, dict) and "q_online" in ckpt:
    model.load_state_dict(ckpt["q_online"])
else:
    model.load_state_dict(ckpt)
model.eval()
print(f"Modelo cargado: {MODEL_PATH}", flush=True)


# ==========================================
# Expansion recursiva del arbol
# ==========================================

def expand_tree(state, depth, branch_probability, branch_prob_local, history):

    purity = float(np.max(state))

    # Estado terminal
    if purity >= PURITY_THRESHOLD:
        return {
            "type": "terminal",
            "depth": depth,
            "probability": branch_probability,
            "branch_prob": branch_prob_local,
            "terminal_state": int(np.argmax(state)),
            "purity": purity,
            "history": history,
        }

    # Limite de profundidad (no terminal -> lo marcamos)
    if depth >= MAX_DEPTH:
        return {
            "type": "cutoff",
            "depth": depth,
            "probability": branch_probability,
            "branch_prob": branch_prob_local,
            "purity": purity,
            "history": history,
        }

    # Inferencia greedy
    with torch.no_grad():
        state_t  = torch.as_tensor(state, dtype=torch.float32)
        q_values = model(state_t)
        action   = int(torch.argmax(q_values).item())

    pulse = env.ACTIONS[action]
    A0, A1 = env.get_transition_matrices(action)

    s0 = A0 @ state
    s1 = A1 @ state
    p0 = float(np.sum(s0))
    p1 = float(np.sum(s1))

    if p0 > 1e-12:
        s0 = s0 / p0
    if p1 > 1e-12:
        s1 = s1 / p1

    node = {
        "type": "node",
        "depth": depth,
        "probability": branch_probability,
        "branch_prob": branch_prob_local,
        "action": action + 1,
        "pulse_label": pulse["label"],
        "p(k=0)": p0,
        "p(k=1)": p1,
        "purity": purity,
        "history": history,
        "children": {},
    }

    if p0 > 1e-8:
        child0 = expand_tree(
            s0, depth + 1, branch_probability * p0, p0,
            history + [{"action": action + 1, "measurement": "k=0", "probability": p0}]
        )
        if child0:
            node["children"]["k=0"] = child0

    if p1 > 1e-8:
        child1 = expand_tree(
            s1, depth + 1, branch_probability * p1, p1,
            history + [{"action": action + 1, "measurement": "k=1", "probability": p1}]
        )
        if child1:
            node["children"]["k=1"] = child1

    return node


# ==========================================
# Estado inicial FIJO (determinista)
# Mezcla uniforme J=1 -> arbol reproducible
# ==========================================

initial_state = np.zeros(env.n_states, dtype=np.float32)
for i in range(6):
    initial_state[i] = 1.0 / 6.0

print("Construyendo arbol...", flush=True)
tree_root = expand_tree(
    state=initial_state,
    depth=0,
    branch_probability=1.0,
    branch_prob_local=1.0,
    history=[],
)

with open(JSON_PATH, "w") as f:
    json.dump(tree_root, f, indent=2)

print(f"Arbol guardado: {JSON_PATH}", flush=True)