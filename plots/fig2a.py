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
from matplotlib.lines import Line2D


# ==========================================
# CaH+ Energy Level Diagram (Fig. 2a)
# ==========================================
#
# States labeled I-XVI
# J=1: states I-VI    (6 states)
# J=2: states VII-XVI (10 states)
#
# Quantum numbers: |J, m, xi>
# xi = + or -
# ==========================================

# ==========================================
# State definitions
# ==========================================

# J=1 manifold (m = -3/2, -1/2, 1/2, 3/2)
# xi = + and -

J1_states = [
    # (label, m,    xi, x_pos)
    ("I",    -3/2, "+", 0),
    ("II",   -1/2, "+", 1),
    ("III",   1/2, "+", 2),
    ("IV",  -3/2,  "-", 0),
    ("V",   -1/2,  "-", 1),
    ("VI",   1/2,  "-", 2),
]

# J=2 manifold (m = -5/2, -3/2, -1/2, 1/2, 3/2, 5/2)
# xi = + and -

J2_states = [
    # (label, m,    xi, x_pos)
    ("VII",  -3/2, "+", 0),
    ("VIII", -1/2, "+", 1),
    ("IX",    1/2, "+", 2),
    ("X",     3/2, "+", 3),
    ("XI",    5/2, "+", 4),
    ("XII",  -5/2, "-", -1),
    ("XIII", -3/2, "-", 0),
    ("XIV",  -1/2, "-", 1),
    ("XV",    1/2, "-", 2),
    ("XVI",   3/2, "-", 3),
]

# ==========================================
# Energy positions (arbitrary units)
# scaled to match paper visual
# ==========================================

# J=1 base energy
E_J1 = 0.0

# J=2 base energy (0.57 THz above J=1)
E_J2 = 4.0

# Zeeman splitting within each manifold
# 26.1 kHz for J=1, 37.6 kHz for J=2
# (scaled for visualization)

zeeman_J1 = 0.12   # scaled
zeeman_J2 = 0.15   # scaled

# xi split (+ above -, small offset)
xi_split = 0.05


def get_energy(J, m, xi):
    if J == 1:
        E = E_J1 + m * zeeman_J1
    else:
        E = E_J2 + m * zeeman_J2

    if xi == "+":
        E += xi_split
    else:
        E -= xi_split

    return E


# ==========================================
# Build state positions
# ==========================================

state_positions = {}

for label, m, xi, xpos in J1_states:
    E = get_energy(1, m, xi)
    state_positions[label] = {
        "m": m, "xi": xi,
        "E": E, "J": 1,
        "x": m
    }

for label, m, xi, xpos in J2_states:
    E = get_energy(2, m, xi)
    state_positions[label] = {
        "m": m, "xi": xi,
        "E": E, "J": 2,
        "x": m
    }

# ==========================================
# Pulse definitions (Table S2)
# Red arrows 1-9: concentrate population
# Blue arrows 10-13: cross-J transitions
# ==========================================

# Format: (from_label, to_label, pulse_num)

red_pulses = [
    ("I",    "VII",   1),
    ("II",   "VIII",  2),
    ("III",  "IX",    3),
    ("IV",   "VII",   4),
    ("XII",  "XII",   5),   # self (flip spin)
    ("IV",   "V",     6),
    ("X",    "X",     7),   # self (flip spin)
    ("XI",   "XI",    8),   # self (flip spin)
    ("VI",   "XI",    9),
]

blue_pulses = [
    ("IV",   "V",    10),
    ("V",    "IV",   11),
    ("XII",  "XIII", 12),
    ("XIII", "XII",  13),
]

# ==========================================
# Plot
# ==========================================

fig, ax = plt.subplots(figsize=(7, 6))

level_len = 0.35

# ------------------------------------------
# Draw energy levels
# ------------------------------------------

for label, info in state_positions.items():

    x   = info["x"]
    E   = info["E"]
    xi  = info["xi"]

    color = "black"

    ax.plot(
        [x - level_len/2, x + level_len/2],
        [E, E],
        color=color,
        linewidth=1.5,
        solid_capstyle="round"
    )

    ax.text(
        x,
        E + 0.04,
        label,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="black"
    )

# ------------------------------------------
# Draw red arrows (pulses 1-9)
# ------------------------------------------

for (from_l, to_l, pnum) in red_pulses:

    if from_l not in state_positions:
        continue
    if to_l not in state_positions:
        continue

    x0 = state_positions[from_l]["x"]
    y0 = state_positions[from_l]["E"]
    x1 = state_positions[to_l]["x"]
    y1 = state_positions[to_l]["E"]

    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->",
            color="red",
            lw=1.2,
        )
    )

    # pulse number label
    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2

    ax.text(
        xm + 0.05,
        ym,
        str(pnum),
        fontsize=6,
        color="red",
        ha="left",
        va="center"
    )

# ------------------------------------------
# Draw blue arrows (pulses 10-13)
# ------------------------------------------

for (from_l, to_l, pnum) in blue_pulses:

    if from_l not in state_positions:
        continue
    if to_l not in state_positions:
        continue

    x0 = state_positions[from_l]["x"]
    y0 = state_positions[from_l]["E"]
    x1 = state_positions[to_l]["x"]
    y1 = state_positions[to_l]["E"]

    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="<->",
            color="blue",
            lw=1.2,
        )
    )

    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2

    ax.text(
        xm + 0.05,
        ym,
        str(pnum),
        fontsize=6,
        color="blue",
        ha="left",
        va="center"
    )

# ------------------------------------------
# xi labels (+/-)
# ------------------------------------------

ax.text(
    5.5, E_J1 + xi_split,
    "+", fontsize=10,
    color="black", va="center"
)

ax.text(
    5.5, E_J1 - xi_split,
    "−", fontsize=10,
    color="black", va="center"
)

ax.text(
    5.5, E_J2 + xi_split,
    "+", fontsize=10,
    color="black", va="center"
)

ax.text(
    5.5, E_J2 - xi_split,
    "−", fontsize=10,
    color="black", va="center"
)

# ------------------------------------------
# J labels and energy annotations
# ------------------------------------------

ax.text(
    -2.5, E_J1,
    "J=1", fontsize=10,
    fontweight="bold",
    va="center"
)

ax.text(
    -2.5, E_J2,
    "J=2", fontsize=10,
    fontweight="bold",
    va="center"
)

# Energy gap annotation
ax.annotate(
    "",
    xy=(-2.0, E_J2 - 0.3),
    xytext=(-2.0, E_J1 + 0.3),
    arrowprops=dict(
        arrowstyle="<->",
        color="blue",
        lw=1.5
    )
)

ax.text(
    -1.8,
    (E_J1 + E_J2) / 2,
    "0.57 THz",
    fontsize=8,
    color="blue",
    va="center"
)

# Zeeman splitting annotations
ax.annotate(
    "",
    xy=(4.8, E_J2 + zeeman_J2 * 2.5 + xi_split),
    xytext=(4.8, E_J2 - xi_split),
    arrowprops=dict(
        arrowstyle="<->",
        color="black",
        lw=1.0
    )
)

ax.text(
    5.0,
    E_J2 + zeeman_J2 * 1.2,
    "37.6 kHz",
    fontsize=7,
    va="center"
)

ax.annotate(
    "",
    xy=(4.0, E_J1 + zeeman_J1 * 1.5 + xi_split),
    xytext=(4.0, E_J1 - xi_split),
    arrowprops=dict(
        arrowstyle="<->",
        color="black",
        lw=1.0
    )
)

ax.text(
    4.2,
    E_J1 + zeeman_J1 * 0.7,
    "26.1 kHz",
    fontsize=7,
    va="center"
)

# ------------------------------------------
# m axis labels
# ------------------------------------------

ax.set_xlabel("m", fontsize=10)

m_ticks_J1 = [-3/2, -1/2, 1/2, 3/2]
m_ticks_J2 = [-5/2, -3/2, -1/2, 1/2, 3/2, 5/2]

ax.set_xticks(
    sorted(set(m_ticks_J1 + m_ticks_J2))
)

ax.set_xticklabels([
    "−5/2", "−3/2", "−1/2",
    "1/2",  "3/2",  "5/2"
], fontsize=8)

# ------------------------------------------
# B field annotation
# ------------------------------------------

ax.text(
    3.5, E_J2 + 0.8,
    "B = 0.36 mT",
    fontsize=8,
    style="italic"
)

# ------------------------------------------
# Clean up axes
# ------------------------------------------

ax.set_ylabel("E", fontsize=10)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.set_xlim(-3, 6.5)
ax.set_ylim(-0.8, 5.5)

plt.tight_layout()
plt.savefig("fig2a.png", dpi=300)
plt.show()

print("Fig. 2a guardada.", flush=True)