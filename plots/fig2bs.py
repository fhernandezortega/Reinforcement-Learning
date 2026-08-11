"""
fig2b.py — Figura 2b del RL-QLS (entrenamiento), semilla PRIMARIA (normal).
Lee training_history_seed0.json (o el que se indique abajo en SEED).
Genera fig2b.png: episodios individuales (naranja) + promedio movil de 100
(azul) + sweeping protocol (magenta).
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

SEED = 0                      # <- semilla "normal" (primaria por defecto)
OUTNAME = "fig2bs.png"

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def find_history(seed):
    name = f"training_history_seed{seed}.json"
    for p in [os.path.join(ROOT, "rl", name), os.path.join(ROOT, name),
              os.path.join(HERE, name), os.path.join(os.getcwd(), name)]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No encontre '{name}'. Corre train_cah.py (que ahora guarda un "
        f"history por semilla), o ajusta SEED en este script.")


def plot_2b(history_path, out_path):
    with open(history_path) as f:
        h = json.load(f)
    steps = np.array(h["steps"]); N = len(steps); ep = np.arange(1, N + 1)
    sweeping = float(h.get("sweeping", 10.0))
    w = min(100, N)
    avg = np.convolve(steps, np.ones(w) / w, mode="valid")
    aep = np.arange(w, N + 1)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot(ep, steps, color="orange", lw=0.6, alpha=0.6,
            label="RL individual", zorder=1)
    ax.plot(aep, avg, color="blue", lw=1.6, label="RL averaged", zorder=3)
    ax.axhline(sweeping, color="magenta", lw=1.8,
               label="sweeping protocol", zorder=2)
    ax.set_xlabel("# training episodes"); ax.set_ylabel("# steps per episode")
    ax.set_xlim(0, N); ax.set_ylim(0, 22)
    ax.text(0.03, 0.06, "training", transform=ax.transAxes, fontsize=12,
            color="white", fontweight="bold", va="bottom", ha="left",
            bbox=dict(boxstyle="square,pad=0.35", facecolor="royalblue",
                      edgecolor="none"), zorder=4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight")

    n_clip = int((steps > 22).sum())
    print(f"[fig2b] semilla {h.get('seed')}: arranque azul={avg[0]:.2f}, "
          f"convergencia={steps[-100:].mean():.2f}, "
          f"espigas>22={n_clip} ({100*n_clip/N:.1f}%)")
    print(f"[fig2b] guardada: {out_path}")


if __name__ == "__main__":
    hp = find_history(SEED)
    plot_2b(hp, os.path.join(HERE, OUTNAME))