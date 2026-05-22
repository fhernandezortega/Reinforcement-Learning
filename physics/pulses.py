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
from scipy.linalg import expm

from physics.hamiltonian import MolecularHamiltonian


# ==========================================
# Raman pulse model
# ==========================================

class RamanPulse:

    def __init__(self):

        self.ham = MolecularHamiltonian()

        self.H0 = self.ham.build()

        self.n_states = self.ham.n_states

    # ======================================
    # Pulse Hamiltonian
    # ======================================

    def pulse_hamiltonian(
        self,
        i,
        f,
        omega
    ):

        Hp = np.zeros(
            (self.n_states, self.n_states),
            dtype=complex
        )

        Hp[i, f] = omega
        Hp[f, i] = omega

        return Hp

    # ======================================
    # Total Hamiltonian
    # ======================================

    def total_hamiltonian(
        self,
        i,
        f,
        omega
    ):

        Hp = self.pulse_hamiltonian(
            i,
            f,
            omega
        )

        return Hp

    # ======================================
    # Time evolution operator
    # ======================================

    def propagator(
        self,
        i,
        f,
        omega,
        t
    ):

        H = self.total_hamiltonian(
            i,
            f,
            omega
        )

        U = expm(
            -1j * H * t
        )

        return U

    # ======================================
    # Transition matrices for k=0 and k=1
    # (projective measurement, Eq. 4b paper)
    # ======================================

    def transition_matrices(
        self,
        i,
        f,
        omega,
        t
    ):

        """
        Returns A_k=0 and A_k=1 (Eq. 4b)

        A_k=0[j,l] = |U_j0,l0|^2  -> k=0 outcome
        A_k=1[j,l] = |U_j1,l0|^2  -> k=1 outcome

        U is the (n_states*2) x (n_states*2)
        propagator in the |state> x |motional>
        space, Lamb-Dicke regime (k in {0,1})
        """

        U = self.propagator(i, f, omega, t)

        n = self.n_states

        # In Lamb-Dicke regime:
        # u_JJ' = <J',k=0|U|J,k=0>  -> stays in k=0
        # v_JJ' = <J',k=1|U|J,k=0>  -> goes to k=1

        # For our simplified model:
        # The pulse couples states i and f
        # u coefficients: diagonal-like (k=0 outcome)
        # v coefficients: off-diagonal (k=1 outcome)

        # A_k=0[j,l] = |u_lj|^2
        # A_k=1[j,l] = |v_lj|^2

        # u = cos(omega*t) part (stays in k=0)
        # v = -i*sin(omega*t) part (goes to k=1)

        # Build u and v matrices from U
        # u_jl = U_jl for j != target sideband
        # For blue sideband: couples |J,k=0> to |J',k=1>

        # Simplified: use |U|^2 decomposition
        # A0 accounts for population remaining (k=0)
        # A1 accounts for population transferred (k=1)

        # From Eq. S1 of the paper:
        # u_JJ' describes |J,k=0> -> |J',k=0>
        # v_JJ' describes |J,k=0> -> |J',k=1>

        # For our 2-level coupling (i<->f):
        # omega*t = pi/2 is a pi-pulse on the sideband

        theta = omega * t

        # u matrix (k=0 block)
        u = np.eye(n, dtype=complex)
        u[i, i] = np.cos(theta)
        u[f, f] = np.cos(theta)
        u[i, f] = -1j * np.sin(theta)
        u[f, i] = -1j * np.sin(theta)

        # v matrix (k=1 block)
        # In blue-sideband: only the target
        # transition contributes to k=1
        v = np.zeros((n, n), dtype=complex)
        v[f, i] = -1j * np.sin(theta)
        v[i, f] = -1j * np.sin(theta)

        # Transition matrices (population)
        A0 = np.abs(u) ** 2   # k=0 outcome
        A1 = np.abs(v) ** 2   # k=1 outcome

        return A0, A1

    # ======================================
    # Transition matrix (original, k-averaged)
    # ======================================

    def transition_matrix(
        self,
        i,
        f,
        omega,
        t
    ):

        U = self.propagator(i, f, omega, t)

        A = np.abs(U) ** 2

        return A


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    pulse = RamanPulse()

    A0, A1 = pulse.transition_matrices(
        i=0,
        f=1,
        omega=np.pi / 2,
        t=1.0
    )

    print("\nA_k=0:\n", np.round(A0, 3), flush=True)
    print("\nA_k=1:\n", np.round(A1, 3), flush=True)

    state = np.array([1.0, 0.0, 0.0, 0.0])

    p0 = np.sum(A0 @ state)
    p1 = np.sum(A1 @ state)

    print(f"\np(k=0) = {p0:.4f}", flush=True)
    print(f"p(k=1) = {p1:.4f}", flush=True)