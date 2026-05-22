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
from itertools import product


# ==========================================
# CaH+ Molecular Hamiltonian
# Eq. S9 of the paper:
# H = 2pi R J^2 - g muN J.B - gI muN I.B
#     - 2pi cIJ I.J
# Parameters from NIST experiments
# B = 0.36 mT
# ==========================================

class CaHHamiltonian:

    def __init__(self):

        # ==================================
        # Molecular constants (SI units)
        # from Chou et al. Nature 2017
        # ==================================

        # Rotational constant (Hz)
        self.R = 142.5e9

        # Magnetic field (T)
        self.B = 0.36e-3

        # Nuclear magneton (Hz/T)
        self.muN = 7.622593e6

        # Rotational g-factor
        self.g = -0.00040

        # Nuclear g-factor (H nucleus)
        self.gI = 5.585695

        # Spin-rotation constant (Hz)
        self.cIJ = 19.6e3

        # Nuclear spin (H: I=1/2)
        self.I_spin = 0.5

        # Consider J = 1, 2 manifolds
        self.J_max = 2

        # ==================================
        # Build basis states
        # |J, m, xi> where xi = +/-
        # Total: 4*(2*1+1) + 4*(2*2+1) = 16
        # but paper uses |J, mJ, mI, xi>
        # We use coupled basis |J, m, xi>
        # with m = total magnetic quantum number
        # ==================================

        self.basis = self._build_basis()

        self.n_states = len(self.basis)

    # ======================================
    # Build basis states
    # ======================================

    def _build_basis(self):

        """
        Build basis |J, mJ, mI> for J=1,2
        I=1/2 (proton spin)

        Returns list of (J, mJ, mI) tuples
        """

        basis = []

        mI_vals = [-0.5, 0.5]

        for J in [1, 2]:

            mJ_vals = list(range(-J, J + 1))

            for mJ in mJ_vals:
                for mI in mI_vals:

                    basis.append((J, mJ, mI))

        return basis

    # ======================================
    # Matrix elements
    # ======================================

    def _build_matrix(self):

        n = self.n_states

        H = np.zeros((n, n), dtype=complex)

        for idx, (J, mJ, mI) in enumerate(self.basis):

            # ==========================
            # Diagonal: rotational term
            # E_rot = 2pi R J(J+1)
            # ==========================

            E_rot = 2 * np.pi * self.R * J * (J + 1)

            # ==========================
            # Diagonal: Zeeman terms
            # E_Z = -g muN mJ B
            #      - gI muN mI B
            # ==========================

            E_Z = (
                - self.g * self.muN * mJ * self.B
                - self.gI * self.muN * mI * self.B
            )

            # ==========================
            # Diagonal: spin-rotation
            # E_sr = -2pi cIJ mI mJ
            # (diagonal part)
            # ==========================

            E_sr = -2 * np.pi * self.cIJ * mI * mJ

            H[idx, idx] = E_rot + E_Z + E_sr

        return H

    # ======================================
    # Diagonalize
    # ======================================

    def build(self):

        return self._build_matrix()

    def diagonalize(self):

        H = self.build()

        energies, states = np.linalg.eigh(H)

        return energies, states

    # ======================================
    # Get state labels
    # ======================================

    def get_labels(self):

        labels = []

        for (J, mJ, mI) in self.basis:

            labels.append(
                f"|{J},{mJ},{mI:+.1f}>"
            )

        return labels


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    ham = CaHHamiltonian()

    print(f"\nNumber of states: {ham.n_states}", flush=True)

    print("\nBasis states:", flush=True)

    for i, label in enumerate(ham.get_labels()):
        print(f"  {i:2d}: {label}", flush=True)

    H = ham.build()

    energies, _ = ham.diagonalize()

    print("\nDiagonal energies (GHz):", flush=True)

    for i, E in enumerate(np.diag(H).real):
        print(
            f"  {i:2d}: {E/1e9:.6f} GHz  "
            f"{ham.get_labels()[i]}",
            flush=True
        )