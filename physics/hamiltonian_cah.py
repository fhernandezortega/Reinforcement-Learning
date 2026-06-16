import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import numpy as np
import qutip as qt
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────
# Angular momentum operators
# ─────────────────────────────────────────────────────────────────────

def Jz_op(J):
    dim  = int(2*J + 1)
    diag = np.array([J - k for k in range(dim)], dtype=float)
    return qt.Qobj(np.diag(diag))

def Jp_op(J):
    dim     = int(2*J + 1)
    mJ_vals = np.array([J - k for k in range(dim)])
    mat     = np.zeros((dim, dim), dtype=complex)
    for col, mJ in enumerate(mJ_vals):
        row = col - 1
        if 0 <= row < dim:
            mat[row, col] = np.sqrt(J*(J+1) - mJ*(mJ+1))
    return qt.Qobj(mat)

def Jm_op(J): return Jp_op(J).dag()
def Jx_op(J): return 0.5 * (Jp_op(J) + Jm_op(J))
def Jy_op(J): return -0.5j * (Jp_op(J) - Jm_op(J))

# Nuclear spin I=1/2
Iz = 0.5 * qt.sigmaz()
Ip = qt.sigmap()
Im = qt.sigmam()
Ix = 0.5 * qt.sigmax()
Iy = 0.5 * qt.sigmay()


# =====================================================================
# MolecularConstants
# =====================================================================

@dataclass
class MolecularConstants:
    """
    Constantes moleculares de CaH+ (Chou et al., Nature 2017).
    Importable por env/rlqls_env_cah.py y physics/pulses_cah.py.
    """
    R:      float = 142.5e9    # constante rotacional (Hz)
    g:      float = -0.00040   # g rotacional
    gI:     float = 5.585695   # g nuclear (proton)
    cIJ:    float = 19.6e3     # spin-rotacion (Hz)
    muN:    float = 7.622593e6 # magneton nuclear (Hz/T)
    B:      float = 0.36e-3    # campo magnetico (T)
    I_spin: float = 0.5
    J_min:  int   = 1
    J_max:  int   = 2


# =====================================================================
# CaHHamiltonian — implementacion principal
# =====================================================================

class CaHHamiltonian:
    """
    Hamiltoniano completo de CaH+ con acoplamiento I·J (Ec. S9).

      H = 2π R J² − g µN J·B − gI µN I·B − 2π cIJ I·J

    Base no acoplada: |J, mJ, mI⟩
    16 estados para J in {1, 2}.
    """

    # Constantes por defecto (se sobreescriben si se pasa MolecularConstants)
    R      = 142.5e9
    g      = -0.00040
    gI     = 5.585695
    cIJ    = 19.6e3
    muN    = 7.622593e6
    B      = 0.36e-3
    I_spin = 0.5

    def __init__(self, constants: Optional[MolecularConstants] = None):
        if constants is not None:
            self.R           = constants.R
            self.g           = constants.g
            self.gI          = constants.gI
            self.cIJ         = constants.cIJ
            self.muN         = constants.muN
            self.B           = constants.B
            self.J_manifolds = list(range(constants.J_min,
                                          constants.J_max + 1))
        else:
            self.J_manifolds = [1, 2]

        self._build_basis()
        self._build_hamiltonian()

    # ─── Basis ───────────────────────────────────────────────────────

    def _build_basis(self):
        self.basis: List[Tuple[int, int, float]] = []
        for J in self.J_manifolds:
            for mJ in range(-J, J + 1):
                for mI in [-0.5, 0.5]:
                    self.basis.append((J, mJ, mI))
        self.n_states = len(self.basis)

    # ─── Hamiltonian ──────────────────────────────────────────────────

    def _build_hamiltonian(self):
        blocks = {}
        for J in self.J_manifolds:
            dim_J = int(2*J + 1)

            Jz_ = qt.tensor(Jz_op(J), qt.qeye(2))
            Jp_ = qt.tensor(Jp_op(J), qt.qeye(2))
            Jm_ = qt.tensor(Jm_op(J), qt.qeye(2))

            _Iz = qt.tensor(qt.qeye(dim_J), Iz)
            _Ip = qt.tensor(qt.qeye(dim_J), Ip)
            _Im = qt.tensor(qt.qeye(dim_J), Im)

            IJ = _Iz * Jz_ + 0.5 * (_Ip * Jm_ + _Im * Jp_)

            H_J = (2*np.pi * self.R * J*(J+1)
                   * qt.tensor(qt.qeye(dim_J), qt.qeye(2))
                   - self.g  * self.muN * self.B * Jz_ * 2*np.pi
                   - self.gI * self.muN * self.B * _Iz * 2*np.pi
                   - 2*np.pi * self.cIJ * IJ)

            blocks[J] = H_J

        H_full = np.zeros((self.n_states, self.n_states), dtype=complex)

        def gidx(J, mJ, mI):
            return self.basis.index((J, mJ, mI))

        for J, H_J in blocks.items():
            mat     = H_J.full()
            mJ_vals = list(range(-J, J+1))
            mI_vals = [-0.5, 0.5]
            for mJ_i, mJ in enumerate(mJ_vals):
                for mI_i, mI in enumerate(mI_vals):
                    row_l = mJ_i*2 + mI_i
                    g_row = gidx(J, mJ, mI)
                    for mJ_j, mJp in enumerate(mJ_vals):
                        for mI_j, mIp in enumerate(mI_vals):
                            col_l = mJ_j*2 + mI_j
                            g_col = gidx(J, mJp, mIp)
                            H_full[g_row, g_col] = mat[row_l, col_l]

        self.H_matrix = H_full
        self.H_qobj   = qt.Qobj(H_full)

    # ─── API original (sin cambios) ───────────────────────────────────

    def build(self):
        """Devuelve H_matrix (compatibilidad con codigo original)."""
        return self.H_matrix

    def diagonalize(self):
        energies, eigvecs = np.linalg.eigh(self.H_matrix)
        self.energies = energies
        self.eigvecs  = eigvecs
        return energies, eigvecs

    def get_labels(self):
        return [f"|J={J}, mJ={mJ:+d}, mI={mI:+.1f}>"
                for (J, mJ, mI) in self.basis]

    def get_boltzmann(self, T_K: float = 300.0) -> np.ndarray:
        """Poblacion termica en base de eigenstados (API original)."""
        energies, _ = self.diagonalize()
        kB   = 1.380649e-23
        hbar = 1.054571817e-34
        E_J  = energies * hbar
        beta = 1.0 / (kB * T_K)
        w    = np.exp(-beta * E_J)
        return w / w.sum()

    # ─── API nueva requerida por rlqls_env_cah.py ─────────────────────

    def boltzmann_population(self, T_bbr: float = 300.0,
                              use_eigenstates: bool = True) -> np.ndarray:
        """Alias de get_boltzmann() con la firma del entorno RL."""
        return self.get_boltzmann(T_K=T_bbr)

    def state_index(self, J: int, mJ: int, mI: float) -> int:
        return self.basis.index((J, mJ, float(mI)))

    def __repr__(self):
        return (f"CaHHamiltonian(J={self.J_manifolds[0]}..{self.J_manifolds[-1]}, "
                f"n_states={self.n_states})")


# =====================================================================
# CaHPlusHamiltonian — alias directo de CaHHamiltonian
# Permite importar con ambos nombres sin duplicar codigo.
# =====================================================================

CaHPlusHamiltonian = CaHHamiltonian


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # API original
    ham = CaHHamiltonian()
    print(f"{ham}")
    P = ham.get_boltzmann(T_K=300.0)
    print(f"get_boltzmann sum = {P.sum():.6f}")

    # API nueva via alias
    ham2 = CaHPlusHamiltonian(MolecularConstants(J_max=2))
    ham2.diagonalize()
    P2 = ham2.boltzmann_population(T_bbr=300.0)
    print(f"boltzmann_population sum = {P2.sum():.6f}")

    # Verificar que son el mismo objeto
    assert CaHPlusHamiltonian is CaHHamiltonian
    print("CaHPlusHamiltonian is CaHHamiltonian ✓")
    print("Todos los imports funcionaran correctamente.")