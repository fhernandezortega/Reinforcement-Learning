"""
Genera la curva acumulativa

% de episodios terminados
vs
número de pulsos aplicados

igual a la Fig. S7 del artículo.
"""

import numpy as np
import matplotlib.pyplot as plt


##############################################################################
# Cargar datos
##############################################################################

# Archivo generado durante la evaluación
# Una fila = un episodio
# Valor = número de pulsos utilizados hasta terminar
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

episode_lengths = np.load(os.path.join(ROOT, "episode_lengths.npy"))


print(episode_lengths)
print(episode_lengths.shape)
print(episode_lengths.dtype)
print(len(episode_lengths))

##############################################################################
# Curva acumulativa
##############################################################################

max_pulses = episode_lengths.max()

pulses = np.arange(1, max_pulses + 1)

success = []

for p in pulses:

    finished = np.sum(episode_lengths <= p)

    success.append(
        100.0 * finished / len(episode_lengths)
    )

success = np.asarray(success)

##############################################################################
# Umbral
##############################################################################

threshold = 85

idx = np.argmax(success >= threshold)

pulse85 = pulses[idx]
success85 = success[idx]

##############################################################################
# Figura
##############################################################################

plt.figure(figsize=(6,5))

plt.plot(
    pulses,
    success,
    lw=2,
    color="blue",
    label="RL"
)

plt.axhline(
    threshold,
    ls="--",
    color="blue",
    alpha=0.6
)

plt.axvline(
    pulse85,
    ymax=success85/100,
    ls="--",
    color="blue",
    alpha=0.6
)

plt.xlim(0, max_pulses)

plt.ylim(0,100)

plt.xlabel("# pulses applied")

plt.ylabel("% of episodes finished")

plt.legend()

plt.tight_layout()

plt.show()