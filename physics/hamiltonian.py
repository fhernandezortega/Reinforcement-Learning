import numpy as np

# ==========================================
# Simple rotational Hamiltonian
# ==========================================

class MolecularHamiltonian:

    def __init__(self):

        # rotational constant
        self.R = 1.0

        # magnetic field
        self.B = 0.1

        # g-factor
        self.g = 0.05

        # define basis size
        self.n_states = 4

    # ======================================
    # Construct Hamiltonian
    # ======================================

    def build(self):

        """
        Simple diagonal Hamiltonian

        H = 2π R J² - g B m

        simplified version of Eq. S9
        """

        H = np.zeros(
            (self.n_states, self.n_states)
        )

        # fake J,m quantum numbers
        quantum_numbers = [
            (0, 0),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for i, (J, m) in enumerate(quantum_numbers):

            energy = (
                2 * np.pi * self.R * J * (J + 1)
                - self.g * self.B * m
            )

            H[i, i] = energy

        return H

    # ======================================
    # Diagonalize Hamiltonian
    # ======================================

    def diagonalize(self):

        H = self.build()

        energies, states = np.linalg.eigh(H)

        return energies, states


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    ham = MolecularHamiltonian()

    H = ham.build()

    print("\nHamiltonian:\n")
    print(H)

    energies, states = ham.diagonalize()

    print("\nEigenenergies:\n")
    print(energies)