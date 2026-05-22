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

from physics.hamiltonian_cah import CaHHamiltonian


# ==========================================
# Raman pulses for CaH+
# Blue-sideband pi-pulses
# Parameters from Table S2 of the paper
# ==========================================

class CaHPulse:

    def __init__(self):

        self.ham = CaHHamiltonian()

        self.H0 = self.ham.build()

        self.n_states = self.ham.n_states

        self.basis = self.ham.basis

        # ==================================
        # Build pulse library
        # Table S2: 13 pulses
        # ==================================

        self.pulse_library = self._build_pulse_library()

    # ======================================
    # Find state index
    # ======================================

    def _find_state(self, J, mJ, mI):

        for idx, (Jb, mJb, mIb) in enumerate(self.basis):

            if Jb == J and mJb == mJ and abs(mIb - mI) < 1e-6:

                return idx

        raise ValueError(
            f"State |{J},{mJ},{mI}> not found"
        )

    # ======================================
    # Build pulse library (Table S2)
    # ======================================

    def _build_pulse_library(self):

        pulses = []

        # Lamb-Dicke parameter
        lambda_LD = 0.09

        pulse_data = [

            # Pulse 1
            (1, -1, -0.5, 2, -2, 0.5, 2.156, 16.2),

            # Pulse 2
            (1, -1, 0.5, 2, -2, -0.5, 1.008, 34.6),

            # Pulse 3
            (1, 0, -0.5, 2, -1, 0.5, 0.621, 52.6),

            # Pulse 4
            (1, 0, 0.5, 2, -1, -0.5, 1.881, 18.7),

            # Pulse 5
            (2, -2, -0.5, 2, -2, 0.5, 1.223, 28.5),

            # Pulse 6
            (1, -1, -0.5, 1, -1, 0.5, 1.174, 29.7),

            # Pulse 7
            (2, 1, -0.5, 2, 1, 0.5, 2.097, 16.6),

            # Pulse 8
            (2, 2, -0.5, 2, 2, 0.5, 0.621, 56.2),

            # Pulse 9
            (1, 1, -0.5, 2, 0, 0.5, 1.221, 23.7),

            # Pulse 10
            (1, -1, -0.5, 1, 0, 0.5, 2.078, 16.8),

            # Pulse 11
            (1, 0, -0.5, 1, -1, 0.5, 2.078, 16.8),

            # Pulse 12
            (2, -2, -0.5, 2, -1, 0.5, 1.852, 18.8),

            # Pulse 13
            (2, -1, -0.5, 2, -2, 0.5, 1.852, 18.8),

        ]

        for (Ji, mJi, mIi, Jf, mJf, mIf,
             omega_kHz, t_ms) in pulse_data:

            try:

                i = self._find_state(Ji, mJi, mIi)
                f = self._find_state(Jf, mJf, mIf)

                # omega in rad/s
                omega = (
                    2 * np.pi * omega_kHz * 1e3
                    * lambda_LD
                )

                # t in seconds
                t = t_ms * 1e-3

                pulses.append({
                    "i": i,
                    "f": f,
                    "omega": omega,
                    "t": t,
                    "label": (
                        f"|{Ji},{mJi},{mIi:+.1f}>"
                        f"->|{Jf},{mJf},{mIf:+.1f}>"
                    )
                })

            except ValueError as e:

                print(f"Warning: {e}", flush=True)

        return pulses

    # ======================================
    # Pulse Hamiltonian
    # ======================================

    def pulse_hamiltonian(self, i, f, omega):

        Hp = np.zeros(
            (self.n_states, self.n_states),
            dtype=complex
        )

        Hp[i, f] = omega
        Hp[f, i] = omega

        return Hp

    # ======================================
    # Propagator U = exp(-iHt)
    # ======================================

    def propagator(self, i, f, omega, t):

        H = self.pulse_hamiltonian(i, f, omega)

        U = expm(-1j * H * t)

        return U

    # ======================================
    # Transition matrices A_k=0, A_k=1
    # ======================================

    def transition_matrices(self, i, f, omega, t):
        theta = omega * t 
        n = self.n_states

        # ==================================
        # Blue-sideband pulse:
        # |i, k=0> -> cos(theta)|i,k=0>
        #           - i*sin(theta)|f,k=1>
        #
        # A0[j,l] = prob of going to |j,k=0>
        #           given started in |l,k=0>
        # A1[j,l] = prob of going to |j,k=1>
        #           given started in |l,k=0>
        # ==================================

        # A0: diagonal, except state i loses
        # population sin^2(theta) to k=1
        A0 = np.eye(n)
        A0[i, i] = np.cos(theta) ** 2   # stays in k=0
        A0[f, f] = 1.0                  # f unaffected in k=0

        # A1: only state i contributes to k=1
        # -> goes to state f with prob sin^2(theta)
        A1 = np.zeros((n, n))
        A1[f, i] = np.sin(theta) ** 2   # i -> f, k=1

        return A0, A1

    # ======================================
    # Precompute all transition matrices
    # ======================================

    def precompute_matrices(self):

        A0_list = []
        A1_list = []

        for pulse in self.pulse_library:

            A0, A1 = self.transition_matrices(
                i=pulse["i"],
                f=pulse["f"],
                omega=pulse["omega"],
                t=pulse["t"]
            )

            A0_list.append(A0)
            A1_list.append(A1)

        return A0_list, A1_list


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    pulse = CaHPulse()

    print(
        f"\nPulse library: "
        f"{len(pulse.pulse_library)} pulses",
        flush=True
    )

    for idx, p in enumerate(pulse.pulse_library):

        print(
            f"  Pulse {idx+1:2d}: {p['label']} "
            f"omega={p['omega']/(2*np.pi*1e3):.3f} kHz "
            f"t={p['t']*1e3:.1f} ms",
            flush=True
        )

    print("\nTesting pulse 1:", flush=True)

    p = pulse.pulse_library[0]

    A0, A1 = pulse.transition_matrices(
        i=p["i"],
        f=p["f"],
        omega=p["omega"],
        t=p["t"]
    )

    state = np.zeros(pulse.n_states)
    state[p["i"]] = 1.0

    p0 = np.sum(A0 @ state)
    p1 = np.sum(A1 @ state)

    print(
        f"  p(k=0) = {p0:.4f}, "
        f"p(k=1) = {p1:.4f}, "
        f"sum = {p0+p1:.4f}",
        flush=True
    )