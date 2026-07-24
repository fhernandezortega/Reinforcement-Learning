"""
fig2a.py — Diagrama de niveles de CaH+ (J=1,2) con la libreria de 13 pulsos.
Reproduce la Fig. 2a del RL-QLS. Los datos (niveles Y flechas) vienen de la
cadena validada: hamiltonian_cah -> generate_pulses_cah. Nada hardcodeado.

Notas de fisica que fija el diagrama:
  - Dentro de cada J, el bloque xi=- esta ARRIBA del xi=+ (los xi=- tienen
    mayor energia: en J=1, xi=+ ocupa 0-1.4 kHz y xi=- 9.0-26.1 kHz).
    Por eso el orden I..XVI del paper lista primero los xi=+.
  - Todos los pulsos son intra-manifold (ΔJ=0).
  - Los pulsos 3, 4 y 9 son multi-transicion: dos flechas cada uno (una en
    J=1 y otra en J=2) con el mismo numero -> 16 flechas para 13 pulsos.
  - 10/11 y 12/13 son pares inversos entre los mismos dos estados: se curvan
    en sentidos opuestos para que ambas se vean.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics.hamiltonian_cah import CaHHamiltonian, rlqls_effective
from physics.generate_pulses_cah import generate_nist_library

ROMAN = ['I','II','III','IV','V','VI','VII','VIII','IX','X',
         'XI','XII','XIII','XIV','XV','XVI']


def datos_fig2a():
    """(niveles, flechas, anotaciones) desde la cadena validada."""
    ham = CaHHamiltonian(rlqls_effective())
    L, E = ham.eig_labels, ham.energies_hz

    niveles = [{"idx": i, "roman": ROMAN[i], "J": J, "m": m, "xi": x,
                "E_kHz": E[i] / 1e3} for i, (J, m, x) in enumerate(L)]

    flechas = []
    for p in generate_nist_library(ham=ham):
        for (li, lf, Om) in p["trans"]:
            flechas.append({"pulso": p["paper_id"],
                            "i": L.index(li), "f": L.index(lf),
                            "Omega_kHz": Om})

    # anotaciones: ancho total de cada manifold y separacion rotacional
    anot = {}
    for J in (1, 2):
        idx = [i for i, (Jl, m, x) in enumerate(L) if Jl == J]
        anot[J] = (max(E[i] for i in idx) - min(E[i] for i in idx)) / 1e3
    anot["dJ_THz"] = (E[L.index((2, -1.5, '+'))]
                      - E[L.index((1, -0.5, '+'))]) / 1e12
    return niveles, flechas, anot


# --------------------------------------------------------------------------
niveles, flechas, anot = datos_fig2a()

# posiciones: x = m ; y = fila (J, xi).  xi=- ARRIBA de xi=+ (mayor energia)
YOF = {(1, '+'): 0.7, (1, '-'): 1.0, (2, '+'): 3.9, (2, '-'): 4.2}
pos = {n["idx"]: (n["m"], YOF[(n["J"], n["xi"])]) for n in niveles}

fig, ax = plt.subplots(figsize=(8, 6))
LL = 0.34

for n in niveles:
    x, y = pos[n["idx"]]
    ax.plot([x - LL/2, x + LL/2], [y, y], color="black", lw=2,
            solid_capstyle="round", zorder=3)
    ax.text(x, y + 0.07, n["roman"], ha="center", va="bottom", fontsize=6)


def arrow(a, b, color, num, rad=0.0):
    x0, y0 = pos[a]; x1, y1 = pos[b]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                shrinkA=5, shrinkB=5,
                                connectionstyle=f"arc3,rad={rad}"), zorder=2)
    # etiqueta desplazada segun la curvatura, para que no pise la flecha
    xm, ym = (x0 + x1)/2, (y0 + y1)/2
    ax.text(xm, ym + (0.055 if rad >= 0 else -0.055), str(num), color=color,
            fontsize=6, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none",
                      alpha=0.85), zorder=4)


for fl in flechas:
    n = fl["pulso"]
    color = "red" if n <= 9 else "blue"
    same_row = pos[fl["i"]][1] == pos[fl["f"]][1]
    if n in (11, 13):          # inversos de 10/12: curvar al otro lado
        rad = -0.35
    elif n in (10, 12):
        rad = 0.35
    elif same_row:
        rad = 0.18
    else:
        rad = 0.0              # cruces xi=+ -> xi=- (pulsos 5 y 6)
    arrow(fl["i"], fl["f"], color, n, rad=rad)

# ---- anotaciones fisicas (valores calculados, no hardcodeados) ----
ax.text(-3.0, np.mean([YOF[(1, '+')], YOF[(1, '-')]]), "J=1",
        fontsize=11, fontweight="bold", ha="right", va="center")
ax.text(-3.0, np.mean([YOF[(2, '+')], YOF[(2, '-')]]), "J=2",
        fontsize=11, fontweight="bold", ha="right", va="center")
ax.annotate("", xy=(-2.6, YOF[(2, '+')] - 0.05), xytext=(-2.6, YOF[(1, '-')] + 0.05),
            arrowprops=dict(arrowstyle="<->", lw=1.2))
ax.text(-2.5, np.mean([YOF[(1, '-')], YOF[(2, '+')]]),
        f"{anot['dJ_THz']:.2f} THz", fontsize=8, va="center")

for xz, J in [(3.2, 1), (3.2, 2)]:
    yp, ym = YOF[(J, '-')], YOF[(J, '+')]
    ax.annotate("", xy=(xz, yp), xytext=(xz, ym),
                arrowprops=dict(arrowstyle="<->", lw=1))
    ax.text(xz + 0.1, (yp + ym)/2, f"{anot[J]:.1f} kHz", fontsize=7, va="center")

for J in (1, 2):
    ax.text(3.0, YOF[(J, '-')], "−", fontsize=11, va="center")
    ax.text(3.0, YOF[(J, '+')], "+", fontsize=11, va="center")
ax.text(1.2, 4.9, "B = 0.357 mT", fontsize=9, style="italic")

ax.set_xticks([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
ax.set_xticklabels(["−5/2", "−3/2", "−1/2", "1/2", "3/2", "5/2"])
ax.set_xlabel("m"); ax.set_ylabel("E (not to scale)"); ax.set_yticks([])
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
ax.set_xlim(-3.4, 3.8); ax.set_ylim(0.2, 5.1)
plt.tight_layout(); plt.savefig("fig2a.png", dpi=150, bbox_inches="tight")
print(f"estados: {len(niveles)} | flechas: {len(flechas)} "
      f"(13 pulsos, 3 multi-transicion)")
print(f"anotaciones calculadas: J=1 {anot[1]:.2f} kHz, J=2 {anot[2]:.2f} kHz, "
      f"ΔJ {anot['dJ_THz']:.3f} THz")