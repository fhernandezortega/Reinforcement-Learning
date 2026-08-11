import os, json
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
def find_file(name):
    for d in (HERE, os.path.abspath(os.path.join(HERE, "..")), os.getcwd()):
        p = os.path.join(d, name)
        if os.path.exists(p): return p
    return os.path.join(HERE, name)

data = json.load(open(find_file("fig2c_data.json")))
eval_at = np.array(data["eval_at"])
curves  = np.array(data["curves"])          # (5 semillas, n_puntos)
median  = np.median(curves, axis=0)
p25     = np.percentile(curves, 25, axis=0)
p75     = np.percentile(curves, 75, axis=0)
inset   = np.array(data["inset_lengths"])
inset_at = data["inset_at"]
sweeping = data["sweeping"]

fig, ax = plt.subplots(figsize=(4.2, 4.0))

# curvas por semilla (verde tenue)
for c in curves:
    ax.plot(eval_at, c, color="green", lw=0.6, alpha=0.30, zorder=1)
# banda +/- std entre semillas
#ax.fill_between(eval_at, mean - std, mean + std, color="green", alpha=0.15, zorder=2)
ax.fill_between(eval_at, p25, p75, color="green", alpha=0.15, zorder=2)
# curva verde promedio (testing)
ax.plot(eval_at, median, color="green", lw=2.2, label="RL testing", zorder=4)
#ax.plot(eval_at, mean, color="green", lw=2.2, label="RL testing", zorder=4)
# sweeping
ax.axhline(sweeping, color="magenta", lw=1.8, label="sweeping protocol", zorder=3)

ax.set_xlabel("# training episodes")
ax.set_ylabel("# steps per episode")
ax.set_xlim(0, eval_at.max())
ax.set_ylim(0,28)
#ax.set_ylim(0, max(30, mean.max() * 1.1))
ax.text(0.03, 0.06, "testing", transform=ax.transAxes,
        fontsize=12, color="white", fontweight="bold", va="bottom", ha="left",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="green", edgecolor="none"),
        zorder=5)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

# inset: proceso de testing del modelo entrenado con inset_at episodios
axin = ax.inset_axes([0.46, 0.46, 0.50, 0.46])
axin.plot(np.arange(len(inset)), inset, color="green", lw=0.4, alpha=0.6)
axin.axhline(inset.mean(), color="black", lw=1.2)
axin.set_title(f"model {inset_at}", fontsize=8)
axin.set_xlabel("# testing episodes", fontsize=7)
axin.set_ylabel("# steps", fontsize=7)
axin.tick_params(labelsize=6)
axin.text(0.95, 0.9, f"mean {inset.mean():.1f}", transform=axin.transAxes,
          fontsize=7, ha="right", va="top")

plt.tight_layout()
out = os.path.join(HERE, "fig2c.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Figura guardada: {out}")