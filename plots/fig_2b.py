import os
import json
import numpy as np
import matplotlib.pyplot as plt


# =====================================
# Rutas (robustas a la estructura del proyecto)
# =====================================

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:            # ejecutado en interprete/notebook
    HERE = os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def find_history(name="training_history.json"):
    """Busca el historial en los sitios habituales; error claro si no aparece."""
    candidates = [
        os.path.join(ROOT, "rl", name),     # rl/ (donde lo guarda train_cah.py)
        os.path.join(ROOT, name),           # raiz del proyecto
        os.path.join(HERE, name),           # junto a este script (plots/)
        os.path.join(os.getcwd(), name),    # directorio actual
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No encontre '{name}'. Busque en:\n  " +
        "\n  ".join(candidates) +
        "\nCorre primero train_cah.py, o exporta RLQLS_OUT a una carpeta fija."
    )


HISTORY_PATH = find_history()
OUT_PATH     = os.path.join(HERE, "fig2b.png")
print(f"Historial: {HISTORY_PATH}")


# =====================================
# Load history
# =====================================

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

steps    = np.array(history["steps"])
episodes = np.arange(1, len(steps) + 1)

# valor del sweeping (usa el guardado si existe, si no 9.7)
sweeping = float(history.get("sweeping", 9.7))


# =====================================
# Moving average (100 episodes)
# =====================================

window = min(100, len(steps))
avg_steps = np.convolve(steps, np.ones(window) / window, mode="valid")
avg_episodes = np.arange(window, len(steps) + 1)


# =====================================
# Plot — estilo paper (Fig. 2b)
# =====================================

fig, ax = plt.subplots(figsize=(4.2, 4.0))

ax.plot(episodes, steps, color="orange", linewidth=0.6, alpha=0.6,
        label="RL individual", zorder=1)
ax.plot(avg_episodes, avg_steps, color="blue", linewidth=1.6,
        label="RL averaged", zorder=3)
ax.axhline(y=sweeping, color="magenta", linewidth=1.8,
           label="sweeping protocol", zorder=2)

ax.set_xlabel("# training episodes")
ax.set_ylabel("# steps per episode")
ax.set_xlim(0, len(steps))
ax.set_ylim(0, 22)

ax.text(0.03, 0.06, "training", transform=ax.transAxes,
        fontsize=12, color="white", fontweight="bold", va="bottom", ha="left",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="royalblue", edgecolor="none"),
        zorder=4)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Figura guardada: {OUT_PATH}", flush=True)
plt.show()