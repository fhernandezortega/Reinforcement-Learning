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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ==========================================
# CaH+ Energy Level Diagram (Fig. 2a)
# ==========================================

fig, ax = plt.subplots(figsize=(8, 6))

# ------------------------------------------
# Energy positions
# ------------------------------------------

E_J1_plus  = 1.0
E_J1_minus = 0.7
E_J2_plus  = 4.2
E_J2_minus = 3.9

level_len  = 0.35
x_spacing  = 1.0

# ------------------------------------------
# J=1 states
# m values: -3/2, -1/2, 1/2, 3/2
# ------------------------------------------

J1_m_vals = [-3/2, -1/2, 1/2, 3/2]

# State labels from paper
# xi=+: I, II, III  (m=-3/2,-1/2,1/2)
# xi=-: IV, V, VI   (m=-3/2,-1/2,1/2)

J1_plus_labels  = {-3/2: "I",   -1/2: "II",  1/2: "III"}
J1_minus_labels = {-3/2: "IV",  -1/2: "V",   1/2: "VI"}

# ------------------------------------------
# J=2 states
# m values: -5/2, -3/2, -1/2, 1/2, 3/2, 5/2
# ------------------------------------------

J2_m_vals = [-5/2, -3/2, -1/2, 1/2, 3/2, 5/2]

J2_plus_labels  = {
    -3/2: "VII", -1/2: "VIII", 1/2: "IX",
     3/2: "X",    5/2: "XI"
}

J2_minus_labels = {
    -5/2: "XII",  -3/2: "XIII", -1/2: "XIV",
     1/2: "XV",    3/2: "XVI"
}

# ------------------------------------------
# Draw levels and collect positions
# ------------------------------------------

positions = {}

def draw_level(ax, m, E, label, color="black"):
    x = m * x_spacing
    ax.plot(
        [x - level_len/2, x + level_len/2],
        [E, E],
        color=color,
        linewidth=2.0,
        solid_capstyle="round",
        zorder=3
    )
    ax.text(
        x, E + 0.06,
        label,
        ha="center", va="bottom",
        fontsize=6, color="black"
    )
    positions[label] = (x, E)

# J=1 xi=+
for m, lbl in J1_plus_labels.items():
    draw_level(ax, m, E_J1_plus, lbl)

# J=1 xi=-
for m, lbl in J1_minus_labels.items():
    draw_level(ax, m, E_J1_minus, lbl)

# J=2 xi=+
for m, lbl in J2_plus_labels.items():
    draw_level(ax, m, E_J2_plus, lbl)

# J=2 xi=-
for m, lbl in J2_minus_labels.items():
    draw_level(ax, m, E_J2_minus, lbl)

# ------------------------------------------
# Draw arrows
# ------------------------------------------

def draw_arrow(ax, from_lbl, to_lbl,
               color, pnum, style="->",
               offset=(0.05, 0)):

    if from_lbl not in positions:
        return
    if to_lbl not in positions:
        return

    x0, y0 = positions[from_lbl]
    x1, y1 = positions[to_lbl]

    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=1.3,
            shrinkA=4,
            shrinkB=4,
        ),
        zorder=2
    )

    xm = (x0 + x1) / 2 + offset[0]
    ym = (y0 + y1) / 2 + offset[1]

    ax.text(
        xm, ym,
        str(pnum),
        fontsize=7,
        color=color,
        ha="left",
        va="center",
        fontweight="bold"
    )

# Red arrows (pulses 1-9): J=1 -> J=2
draw_arrow(ax, "I",   "VII",  "red",  1)
draw_arrow(ax, "II",  "VIII", "red",  2)
draw_arrow(ax, "III", "IX",   "red",  3)
draw_arrow(ax, "IV",  "VII",  "red",  4, offset=(-0.15, 0))
draw_arrow(ax, "XII", "XII",  "red",  5)
draw_arrow(ax, "VI",  "XI",   "red",  9)

# Blue arrows (pulses 10-13): horizontal within J
draw_arrow(ax, "IV",  "V",   "blue", 10, style="<->")
draw_arrow(ax, "V",   "IV",  "blue", 11, style="<->",
           offset=(-0.15, -0.08))
draw_arrow(ax, "XII", "XIII","blue", 12, style="<->")
draw_arrow(ax, "XIII","XII", "blue", 13, style="<->",
           offset=(-0.15, -0.08))

# ------------------------------------------
# xi labels
# ------------------------------------------

x_xi = 3.2 * x_spacing

ax.text(x_xi, E_J1_plus,  "+", fontsize=11,
        va="center", ha="left")
ax.text(x_xi, E_J1_minus, "−", fontsize=11,
        va="center", ha="left")
ax.text(x_xi, E_J2_plus,  "+", fontsize=11,
        va="center", ha="left")
ax.text(x_xi, E_J2_minus, "−", fontsize=11,
        va="center", ha="left")

# ------------------------------------------
# J labels
# ------------------------------------------

ax.text(
    -3.2 * x_spacing, (E_J1_plus + E_J1_minus)/2,
    "J=1", fontsize=11, fontweight="bold",
    va="center", ha="right"
)

ax.text(
    -3.2 * x_spacing, (E_J2_plus + E_J2_minus)/2,
    "J=2", fontsize=11, fontweight="bold",
    va="center", ha="right"
)

# ------------------------------------------
# 0.57 THz gap
# ------------------------------------------

x_gap = -2.8 * x_spacing

ax.annotate(
    "",
    xy=(x_gap, E_J2_minus - 0.1),
    xytext=(x_gap, E_J1_plus + 0.1),
    arrowprops=dict(
        arrowstyle="<->",
        color="black",
        lw=1.2
    )
)

ax.text(
    x_gap + 0.1,
    (E_J1_plus + E_J2_minus) / 2,
    "0.57 THz",
    fontsize=8, va="center"
)

# ------------------------------------------
# Zeeman splitting annotations
# ------------------------------------------

x_zm = 3.8 * x_spacing

# J=2
ax.annotate(
    "",
    xy=(x_zm, E_J2_plus + 0.05),
    xytext=(x_zm, E_J2_minus - 0.05),
    arrowprops=dict(
        arrowstyle="<->",
        color="black", lw=1.0
    )
)

ax.text(
    x_zm + 0.1,
    (E_J2_plus + E_J2_minus)/2,
    "37.6 kHz",
    fontsize=7, va="center"
)

# J=1
ax.annotate(
    "",
    xy=(x_zm, E_J1_plus + 0.05),
    xytext=(x_zm, E_J1_minus - 0.05),
    arrowprops=dict(
        arrowstyle="<->",
        color="black", lw=1.0
    )
)

ax.text(
    x_zm + 0.1,
    (E_J1_plus + E_J1_minus)/2,
    "26.1 kHz",
    fontsize=7, va="center"
)

# ------------------------------------------
# B field
# ------------------------------------------

ax.text(
    1.5 * x_spacing, 5.0,
    "B = 0.36 mT",
    fontsize=9, style="italic"
)

# ------------------------------------------
# m axis
# ------------------------------------------

m_ticks = [-5/2, -3/2, -1/2, 1/2, 3/2, 5/2]

ax.set_xticks([m * x_spacing for m in m_ticks])

ax.set_xticklabels([
    "−5/2", "−3/2", "−1/2",
    "1/2",  "3/2",  "5/2"
], fontsize=9)

ax.set_xlabel("m", fontsize=11)
ax.set_ylabel("E (not to scale)", fontsize=10)
ax.set_yticks([])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.set_xlim(-3.5 * x_spacing, 4.5 * x_spacing)
ax.set_ylim(0.2, 5.2)

plt.tight_layout()
plt.savefig("fig2a.png", dpi=300, bbox_inches="tight")
plt.show()

print("Fig. 2a guardada.", flush=True)