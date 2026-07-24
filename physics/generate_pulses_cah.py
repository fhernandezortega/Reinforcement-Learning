"""
generate_pulses_cah.py — Genera la biblioteca de pulsos DESDE LA FISICA,
sin copiar la Tabla S2. Los 5 pasos:
  1. Diagonalizar H            (hamiltonian_cah.py -> energias, autovectores)
  2. Listar transiciones Δm=±1 intra-J con su frecuencia f = (E_f-E_i)/h
  3. Calcular la tasa Ω        (raman_rates.py, autovectores + 3j + polariz.)
     — |Ω(i→j)| = |Ω(j→i)| (Ec. S10): para Δm=+1 (config π+σ⁺) se usa la
       tasa de la direccion inversa que RamanRates (π+σ⁻) si calcula.
  4. Filtrar (Ω<umbral) y agrupar cuasi-degeneradas (|Δf| < λ_LD·Ω)
  5. Calcular D = π/(λ_LD·Ω_eff): promedio de duraciones π (media armonica
     de las Ω del grupo); si max/min >= 3, la de la transicion lenta (Sec. SC)

Para J∈{1,2} produce un superconjunto de los 13 pulsos de la Tabla S2
(incluye los 3 multi-transicion 3,4,9); generate_nist_library() lo filtra
al esquema NIST exacto. Para sistemas grandes (J≤4, J≤6) generate_pulse_library
produce las bibliotecas de 68/131 pulsos que el paper no tabula.
"""
import sys

import numpy as np
import os

# Añade el directorio padre (raíz del proyecto) al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics.hamiltonian_cah import CaHHamiltonian
from physics.Raman_rates import RamanRates

LAMBDA_LD = 0.09


def generate_pulse_library(ham=None, rates=None, omega_min_khz=0.1,
                           bw_factor=1.0, slow_ratio=3.0, dm=-1):
    """
    Agrupamiento fisico sin parametro 'al ojo':
      - dos transiciones son irresolubles si su separacion en frecuencia es
        menor que el ancho de potencia del pulso, ~ la Rabi de banda lateral
        lambda_LD * Omega  (power broadening / ancho de Fourier ~ 1/D).
        bw_factor=1.0 reproduce exactamente el agrupamiento de la Tabla S2.
      - duracion (Sec. SC del paper): promedio de las duraciones pi
        (= media armonica de las Rabi), salvo si max(O)/min(O) >= slow_ratio,
        caso en que se usa la duracion pi de la transicion lenta (pulso 3).
    """
    ham   = ham or CaHHamiltonian()
    rates = rates or RamanRates(ham)
    rates.calibrate()
    E = ham.energies_hz; L = ham.eig_labels

    # pasos 2-3: transiciones candidatas con f y Omega
    trans = []
    for i, (Ji, mi, _) in enumerate(L):
        for j, (Jj, mj, _) in enumerate(L):
            if Ji == Jj and abs((mj - mi) - dm) < 1e-9:
                f  = (E[j] - E[i]) / 1e3                 # kHz
                # |Ω(i→j)| = |Ω(j→i)| (Ec. S10; Tabla S2: 10/11 y 12/13
                # comparten Ω). RamanRates implementa π+σ⁻ (Δm=-1), asi que
                # para dm=+1 la tasa correcta es la de la direccion inversa.
                Om = max(rates.omega_hz(i, j), rates.omega_hz(j, i)) / 1e3
                if Om >= omega_min_khz:                  # paso 4a: filtrar
                    trans.append(dict(i=i, j=j, f=f, Om=Om))

    # paso 4b: agrupar por irresolubilidad fisica
    trans.sort(key=lambda t: t["f"])
    used = [False] * len(trans)
    pulses = []
    for a in range(len(trans)):
        if used[a]:
            continue
        group = [trans[a]]; used[a] = True
        for b in range(a + 1, len(trans)):
            if used[b]:
                continue
            bw = bw_factor * LAMBDA_LD * max(trans[a]["Om"], trans[b]["Om"])
            if abs(trans[b]["f"] - trans[a]["f"]) < bw:
                group.append(trans[b]); used[b] = True

        # paso 5: f = promedio; D segun regla de la Sec. SC
        f_avg = float(np.mean([g["f"] for g in group]))
        Oms   = [g["Om"] for g in group]
        if max(Oms) / min(Oms) >= slow_ratio:
            Om_eff = min(Oms)                            # excepcion (pulso 3)
        else:
            Om_eff = len(Oms) / sum(1.0 / o for o in Oms)  # media armonica (pulsos 4, 9)
        D_s = np.pi / (LAMBDA_LD * 2 * np.pi * Om_eff * 1e3)
        pulses.append(dict(
            f_kHz=f_avg, D_2pi=D_s * 2 * np.pi * 1e3,
            trans=[(L[g["i"]], L[g["j"]], round(g["Om"], 3)) for g in group]))
    return pulses


# ---------------------------------------------------------------------------
# Curacion J<=2: esquema NIST (Fig. 2a / Tabla S2)
# ---------------------------------------------------------------------------
# Transiciones del esquema NIST, en orden del paper.
# Formato: (num_paper, dm, [((J, m_i, xi_i), (J, m_f, xi_f)), ...])
NIST_13 = [
    ( 1, -1, [((2, +2.5, '+'), (2, +1.5, '+'))]),
    ( 2, -1, [((2, +1.5, '+'), (2, +0.5, '+'))]),
    ( 3, -1, [((1, +1.5, '+'), (1, +0.5, '+')), ((2, +0.5, '+'), (2, -0.5, '+'))]),
    ( 4, -1, [((1, +0.5, '+'), (1, -0.5, '+')), ((2, -0.5, '+'), (2, -1.5, '+'))]),
    ( 5, -1, [((2, -1.5, '+'), (2, -2.5, '-'))]),
    ( 6, +1, [((1, -0.5, '+'), (1, +0.5, '-'))]),
    ( 7, -1, [((2, +1.5, '-'), (2, +0.5, '-'))]),
    ( 8, -1, [((2, +0.5, '-'), (2, -0.5, '-'))]),
    ( 9, -1, [((1, +0.5, '-'), (1, -0.5, '-')), ((2, -0.5, '-'), (2, -1.5, '-'))]),
    (10, +1, [((1, -1.5, '-'), (1, -0.5, '-'))]),
    (11, -1, [((1, -0.5, '-'), (1, -1.5, '-'))]),
    (12, +1, [((2, -2.5, '-'), (2, -1.5, '-'))]),
    (13, -1, [((2, -1.5, '-'), (2, -2.5, '-'))]),
]


def _key(label):
    """Normaliza una etiqueta (J, m, xi) para comparar."""
    J, m, xi = label
    xi = {'+': '+', '-': '-', 1: '+', -1: '-'}.get(xi, str(xi))
    return (int(J), float(m), xi)


def select_nist_13(lib_dm_m1, lib_dm_p1):
    """Filtra las librerias completas (dm=-1 y dm=+1) a los 13 pulsos
    del esquema NIST, en el orden y con la numeracion del paper."""
    libs = {-1: lib_dm_m1, +1: lib_dm_p1}
    out = []
    for n, dm, wanted in NIST_13:
        wset = {(_key(a), _key(b)) for a, b in wanted}
        match = [p for p in libs[dm]
                 if {(_key(li), _key(lf)) for li, lf, _ in p["trans"]} == wset]
        if len(match) != 1:
            raise ValueError(f"pulso paper {n}: {len(match)} candidatos, esperaba 1")
        out.append(dict(match[0], paper_id=n))
    return out


def generate_nist_library(ham=None, rates=None):
    """Las 13 acciones del esquema NIST (Fig. 2a / Tabla S2), en orden
    y con numeracion del paper. Punto de entrada para el RL en J<=2."""
    ham = ham or CaHHamiltonian()
    rates = rates or RamanRates(ham)
    lib_m1 = generate_pulse_library(ham, rates, dm=-1)
    lib_p1 = generate_pulse_library(ham, rates, dm=+1)
    return select_nist_13(lib_m1, lib_p1)


# Referencia: Tabla S2 (f kHz, Ω 2π kHz, D 2π^-1 ms)
TABLA_S2 = {
    1:  (-1.72,  [2.156],        16.2),
    2:  (-1.44,  [1.008],        34.6),
    3:  (-1.03,  [2.138, 0.621], 52.6),
    4:  (-0.23,  [1.881, 1.857], 18.7),
    5:  ( 4.40,  [1.223],        28.5),
    6:  ( 26.13, [1.174],        29.7),
    7:  (-6.12,  [2.097],        16.6),
    8:  (-6.56,  [0.621],        56.2),
    9:  (-7.33,  [1.857, 1.221], 23.7),
    10: ( 9.87,  [2.078],        16.8),
    11: (-9.87,  [2.078],        16.8),
    12: ( 13.13, [1.852],        18.8),
    13: (-13.13, [1.852],        18.8),
}

if __name__ == "__main__":
    from physics.hamiltonian_cah import rlqls_effective
    lib13 = generate_nist_library(ham=CaHHamiltonian(rlqls_effective()))
    print(f"Libreria NIST: {len(lib13)} pulsos\n")
    print(" P   f_mia    f_S2     Δf |  D_mia   D_S2 |  Ω_mias vs Ω_S2 (kHz)")
    for p in lib13:
        n = p["paper_id"]
        f2, O2, D2 = TABLA_S2[n]
        Oms = sorted((float(t[2]) for t in p["trans"]), reverse=True)
        O2s = sorted(O2, reverse=True)
        print(f"{n:2d}  {p['f_kHz']:7.2f} {f2:7.2f} {p['f_kHz']-f2:+6.2f} | "
              f"{p['D_2pi']:6.1f} {D2:6.1f} | {Oms} vs {O2s}")