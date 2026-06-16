"""
CaH+ Pulse Library — Sec. SB / Table S2
=========================================
Dos clases disponibles:

  CaHPulse      — implementacion original con QuTiP TDSE (numerica)
  PulseLibrary  — interfaz simplificada con pulsos pi ideales
                  (requerida por Dynamics.py y rlqls_env_cah.py)

Importaciones validas:
  from physics.pulses_cah import CaHPulse
  from physics.pulses_cah import PulseLibrary
  from physics.pulses_cah import PulseLibrary, Pulse, LAMBDA_LD, NU_MOT_HZ
"""

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import numpy as np
import qutip as qt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Import correcto segun estructura del proyecto
try:
    from physics.hamiltonian_cah import CaHHamiltonian, MolecularConstants
except ImportError:
    from hamiltonian_cah import CaHHamiltonian, MolecularConstants


# ══════════════════════════════════════════════════════════════════════
# Constantes globales (usadas por Dynamics.py)
# ══════════════════════════════════════════════════════════════════════

LAMBDA_LD  = 0.09
NU_MOT_HZ  = 5.164e6
OMEGA_MOT  = 2 * np.pi * NU_MOT_HZ
HBAR       = 1.054571817e-34

# QuTiP 5: opciones como dict (QuTiP 4 usaba qt.Options)
OPTIONS_INTRA = {"atol": 1e-8, "rtol": 1e-6, "nsteps": 100_000}
OPTIONS_INTER = {"atol": 1e-6, "rtol": 1e-4, "nsteps": 100_000}

a_mot  = qt.destroy(2)
ad_mot = qt.create(2)


# ══════════════════════════════════════════════════════════════════════
# Tabla S2
# ══════════════════════════════════════════════════════════════════════

PULSE_TABLE = [
    ( 1,  1, -1, -0.5,  2, -2,  0.5,  -1.72,  2.156, 16.2, "inter"),
    ( 2,  1, -1,  0.5,  2, -2, -0.5,  -1.44,  1.008, 34.6, "inter"),
    ( 3,  1,  0, -0.5,  2, -1,  0.5,  -1.03,  0.621, 52.6, "inter"),
    ( 4,  1,  0,  0.5,  2, -1, -0.5,  -0.23,  1.881, 18.7, "inter"),
    ( 5,  2, -2, -0.5,  2, -2,  0.5,   4.40,  1.223, 28.5, "intra"),
    ( 6,  1, -1, -0.5,  1, -1,  0.5,  26.13,  1.174, 29.7, "intra"),
    ( 7,  2,  1, -0.5,  2,  1,  0.5,  -6.12,  2.097, 16.6, "intra"),
    ( 8,  2,  2, -0.5,  2,  2,  0.5,  -6.56,  0.621, 56.2, "intra"),
    ( 9,  1,  1, -0.5,  2,  0,  0.5,  -7.33,  1.221, 23.7, "inter"),
    (10,  1, -1, -0.5,  1,  0,  0.5,   9.87,  2.078, 16.8, "intra"),
    (11,  1,  0, -0.5,  1, -1,  0.5,  -9.87,  2.078, 16.8, "intra"),
    (12,  2, -2, -0.5,  2, -1,  0.5,  13.13,  1.852, 18.8, "intra"),
    (13,  2, -1, -0.5,  2, -2,  0.5, -13.13,  1.852, 18.8, "intra"),
]

MULTI_COMPONENTS = {
    3: (1, +1, -0.5, 2, -1, +0.5, 2.138),
    9: (1, +1, +0.5, 2, +1, +0.5, 1.857),
}


# ══════════════════════════════════════════════════════════════════════
# Dataclasses requeridas por Dynamics.py
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SingleTransition:
    src_eig:   int
    tgt_eig:   int
    rabi_freq: float
    src_label: str = ""
    tgt_label: str = ""


@dataclass
class Pulse:
    index:       int
    table_id:    Optional[int]
    transitions: List[SingleTransition]
    duration_ms: float
    delta_J:     int
    delta_mJ:    int
    pulse_type:  str = "inter"

    @property
    def n_transitions(self): return len(self.transitions)

    @property
    def label(self):
        main   = self.transitions[0]
        suffix = f" (+{self.n_transitions-1} mas)" if self.n_transitions > 1 else ""
        return (f"Pulse {self.table_id or '?':>2}: "
                f"{main.src_label} -> {main.tgt_label}{suffix}")


# ══════════════════════════════════════════════════════════════════════
# PulseLibrary — interfaz requerida por Dynamics.py y rlqls_env_cah.py
# ══════════════════════════════════════════════════════════════════════

class PulseLibrary:
    """
    Libreria de pulsos pi BSB ideales para CaH+.

    Construye objetos Pulse con SingleTransition compatibles con
    IdealTransitionSolver en Dynamics.py.

    Parameters
    ----------
    ham               : CaHHamiltonian (o CaHPlusHamiltonian, son el mismo)
    use_table_s2_only : True = 13 pulsos Tabla S2 (paper)
                        False = todos los permitidos por reglas E1
    """

    def __init__(self,
                 ham,
                 use_table_s2_only: bool = True):

        self.ham               = ham
        self.use_table_s2_only = use_table_s2_only
        self.pulses: List[Pulse] = []

        # Diagonalizar si no se ha hecho
        if not hasattr(ham, 'energies') or ham.energies is None:
            ham.diagonalize()

        self.eig_energies = ham.energies
        self.eigvecs      = ham.eigvecs

        self._build()

    # ─── Utilidades ──────────────────────────────────────────────────

    def _best_eig(self, J: int, mJ: int, mI: float) -> int:
        """Eigenestado con mayor solapamiento con |J,mJ,mI⟩."""
        try:
            unc = self.ham.basis.index((J, mJ, float(mI)))
        except ValueError:
            raise ValueError(f"Estado |{J},{mJ},{mI}> no encontrado en la base")
        return int(np.argmax(np.abs(self.eigvecs[unc, :])))

    def _eig_label(self, idx: int) -> str:
        dom = int(np.argmax(np.abs(self.eigvecs[:, idx])))
        J, mJ, mI = self.ham.basis[dom]
        return f"|J={J},mJ={mJ:+d},mI={mI:+.1f}>"

    def _duration_ms(self, omega_kHz: float) -> float:
        return (np.pi / (LAMBDA_LD * 2*np.pi*omega_kHz*1e3)) * 1e3

    # ─── Construccion ─────────────────────────────────────────────────

    def _build(self):
        # Agrupar Tabla S2 por table_id
        groups = {}
        for row in PULSE_TABLE:
            tid = row[0]
            groups.setdefault(tid, []).append(row)

        covered = set()
        idx = 0

        for tid in sorted(groups.keys()):
            rows = groups[tid]
            transitions = []
            D_ms    = rows[0][9]
            dJ_main = rows[0][4] - rows[0][1]
            dmJ_main= rows[0][5] - rows[0][2]
            ptype   = "intra" if dJ_main == 0 else "inter"

            for row in rows:
                _, Ji, mJi, mIi, Jf, mJf, mIf, _, Om_kHz, _, _ = row
                try:
                    src = self._best_eig(Ji, mJi, mIi)
                    tgt = self._best_eig(Jf, mJf, mIf)
                except ValueError as e:
                    print(f"  [Aviso] Pulso {tid}: {e}")
                    continue

                Om_rads = 2 * np.pi * Om_kHz * 1e3
                transitions.append(SingleTransition(
                    src_eig=src, tgt_eig=tgt, rabi_freq=Om_rads,
                    src_label=self._eig_label(src),
                    tgt_label=self._eig_label(tgt),
                ))
                covered.add((src, tgt))

            # Agregar componente secundaria si existe
            if tid in MULTI_COMPONENTS:
                Ji2, mJi2, mIi2, Jf2, mJf2, mIf2, Om2_kHz = MULTI_COMPONENTS[tid]
                try:
                    src2 = self._best_eig(Ji2, mJi2, mIi2)
                    tgt2 = self._best_eig(Jf2, mJf2, mIf2)
                    Om2  = 2 * np.pi * Om2_kHz * 1e3
                    transitions.append(SingleTransition(
                        src_eig=src2, tgt_eig=tgt2, rabi_freq=Om2,
                        src_label=self._eig_label(src2),
                        tgt_label=self._eig_label(tgt2),
                    ))
                    covered.add((src2, tgt2))
                except ValueError:
                    pass

            if transitions:
                self.pulses.append(Pulse(
                    index=idx, table_id=tid,
                    transitions=transitions,
                    duration_ms=D_ms,
                    delta_J=dJ_main, delta_mJ=dmJ_main,
                    pulse_type=ptype,
                ))
                idx += 1

        # Auto-enumeracion si se pide
        if not self.use_table_s2_only:
            Om_rads = 2 * np.pi * 0.5e3   # 0.5 kHz por defecto
            D_ms    = self._duration_ms(0.5)
            for i, (Ji, mJi, mIi) in enumerate(self.ham.basis):
                for j, (Jf, mJf, mIf) in enumerate(self.ham.basis):
                    if abs(Jf - Ji) != 1: continue
                    if abs(mJf - mJi) > 1: continue
                    if abs(mIf - mIi) > 1e-9: continue
                    if i == j: continue
                    src = self._best_eig(Ji, mJi, mIi)
                    tgt = self._best_eig(Jf, mJf, mIf)
                    if (src, tgt) in covered: continue
                    dJ  = Jf - Ji
                    dmJ = mJf - mJi
                    self.pulses.append(Pulse(
                        index=idx, table_id=None,
                        transitions=[SingleTransition(
                            src_eig=src, tgt_eig=tgt, rabi_freq=Om_rads,
                            src_label=self._eig_label(src),
                            tgt_label=self._eig_label(tgt),
                        )],
                        duration_ms=D_ms,
                        delta_J=dJ, delta_mJ=dmJ,
                        pulse_type="intra" if dJ == 0 else "inter",
                    ))
                    covered.add((src, tgt))
                    idx += 1

    @property
    def n_actions(self): return len(self.pulses)

    def __repr__(self):
        return (f"PulseLibrary({self.n_actions} pulsos, "
                f"n_states={self.ham.n_states}, "
                f"table_s2_only={self.use_table_s2_only})")


# ══════════════════════════════════════════════════════════════════════
# CaHPulse — implementacion original QuTiP (sin cambios funcionales)
# ══════════════════════════════════════════════════════════════════════

class CaHPulse:
    """Implementacion original con QuTiP TDSE (para verificacion numerica)."""

    def __init__(self, ham: CaHHamiltonian = None):
        if ham is None:
            ham = CaHHamiltonian()
        self.ham      = ham
        self.n_states = ham.n_states
        self.energies, self.eigvecs = ham.diagonalize()
        self._unc_idx = {state: i for i, state in enumerate(ham.basis)}
        self._parse_pulses()

    def _unc_to_eig(self, J, mJ, mI):
        unc_idx  = self._unc_idx[(J, mJ, float(mI))]
        overlaps = np.abs(self.eigvecs[unc_idx, :])
        return int(np.argmax(overlaps))

    def _eig_idx(self, J, mJ, mI):
        return self._unc_to_eig(J, mJ, float(mI))

    def _parse_pulses(self):
        self.pulses = []
        for row in PULSE_TABLE:
            pid, Ji, mJi, mIi, Jf, mJf, mIf, f_kHz, Om_kHz, D_ms, ptype = row
            i_eig = self._eig_idx(Ji, mJi, mIi)
            f_eig = self._eig_idx(Jf, mJf, mIf)
            Om    = 2 * np.pi * Om_kHz * 1e3
            D_s   = D_ms * 1e-3
            transitions = [(i_eig, f_eig, Om)]
            if pid in MULTI_COMPONENTS:
                Ji2, mJi2, mIi2, Jf2, mJf2, mIf2, Om2_kHz = MULTI_COMPONENTS[pid]
                i2  = self._eig_idx(Ji2, mJi2, mIi2)
                f2  = self._eig_idx(Jf2, mJf2, mIf2)
                Om2 = 2 * np.pi * Om2_kHz * 1e3
                transitions.append((i2, f2, Om2))
            self.pulses.append({
                "id": pid, "D_s": D_s, "type": ptype,
                "transitions": transitions,
                "label": f"Pulse {pid:2d}: |{Ji},{mJi:+d},{mIi:+.1f}>->|{Jf},{mJf:+d},{mIf:+.1f}>"
            })

    def precompute_matrices(self, verbose=True):
        A0_list, A1_list = [], []
        for pulse in self.pulses:
            if verbose:
                print(f"  {pulse['label']} ...", end="", flush=True)
            A0, A1 = self._compute_one_TM(pulse)
            A0_list.append(A0)
            A1_list.append(A1)
            if verbose:
                cs = (A0+A1).sum(axis=0)
                print(f" col-sum [{cs.min():.3f},{cs.max():.3f}]")
        return A0_list, A1_list

    def _build_Hint_coeff(self, transitions, t):
        NS = self.n_states
        H_terms = []
        for (i_eig, f_eig, Om) in transitions:
            Ei    = self.energies[i_eig]
            Ef    = self.energies[f_eig]
            delta = Ef - Ei
            fi_op = qt.Qobj(np.outer(np.eye(NS)[f_eig], np.eye(NS)[i_eig]))
            if_op = fi_op.dag()
            H_carrier = qt.tensor(fi_op + if_op, qt.qeye(2))
            H_terms.append([
                H_carrier,
                lambda t, args=None, Om=Om, delta=delta:
                    0.5 * Om * np.cos(delta * t)
            ])
            H_sb_a = qt.tensor(fi_op - if_op, a_mot + ad_mot)
            H_terms.append([
                H_sb_a,
                lambda t, args=None, Om=Om, delta=delta:
                    -LAMBDA_LD * Om / 2.0 * np.sin((delta - OMEGA_MOT) * t)
            ])
        return H_terms

    def _compute_one_TM(self, pulse):
        NS   = self.n_states
        D_s  = pulse["D_s"]
        opts = OPTIONS_INTRA if pulse["type"] == "intra" else OPTIONS_INTER
        H_terms = self._build_Hint_coeff(pulse["transitions"], D_s)
        dt    = 1e-6
        n_pts = max(int(D_s / dt) + 2, 10)
        tlist = np.linspace(0, D_s, n_pts)
        A0 = np.zeros((NS, NS))
        A1 = np.zeros((NS, NS))
        for l in range(NS):
            psi0 = qt.tensor(qt.basis(NS, l), qt.basis(2, 0))
            result = qt.sesolve(H_terms, psi0, tlist, options=opts)
            arr = result.states[-1].full().flatten()
            for j in range(NS):
                A0[j, l] += np.abs(arr[j*2    ])**2
                A1[j, l] += np.abs(arr[j*2 + 1])**2
        return A0, A1


# ══════════════════════════════════════════════════════════════════════
# Quick self-test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Construyendo Hamiltoniano...")
    ham = CaHHamiltonian()
    ham.diagonalize()
    print(f"  {ham.n_states} estados")

    print("\nPulseLibrary (Tabla S2):")
    lib = PulseLibrary(ham, use_table_s2_only=True)
    print(f"  {lib}")
    for p in lib.pulses:
        print(f"    {p.label}  D={p.duration_ms:.1f}ms")

    print("\nPulseLibrary (auto-enumeracion):")
    lib2 = PulseLibrary(ham, use_table_s2_only=False)
    print(f"  {lib2}")