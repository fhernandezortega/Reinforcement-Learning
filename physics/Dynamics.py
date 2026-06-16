"""
dynamics.py — IdealTransitionSolver + ProjectiveMeasurement
============================================================
Referencia: PIPI2026 Ec. S1–S3, Sec. SB

Modelo ideal de pulso pi de banda lateral azul (BSB) en regimen
Lamb-Dicke con k in {0, 1}:

    |i, k=0> --BSB pi--> -i |f, k=1>

Amplitudes de evolucion (Ec. S1):
    u_{j,l} = <j,k=0|U|l,k=0>
    v_{j,l} = <j,k=1|U|l,k=0>

Para pulso pi ideal que mueve |i> -> |f>:
    u_{j,i} = 0 para todo j          (i vacia en k=0)
    v_{j,i} = delta_{j,f}            (f se llena en k=1)
    u_{j,l} = delta_{j,l} para l!=i  (resto permanece en k=0)
    v_{j,l} = 0           para l!=i

Matrices de transicion (Ec. S2-S3):
    A0[j, l] = |u_{j,l}|^2
    A1[j, l] = |v_{j,l}|^2

Probabilidades de medicion (Ec. S2):
    p0 = (A0 @ P).sum()
    p1 = (A1 @ P).sum()

Post-medicion (Ec. S3):
    k=0: P_post = (A0 @ P) / p0
    k=1: P_post = (A1 @ P) / p1

Estocacidad: (A0 + A1).sum(axis=0) = 1 para toda columna.

Para pulsos multi-transicion (pulsos 3, 4, 9 de Tabla S2), dos
transiciones i1->f1 e i2->f2 actuan simultaneamente. Si son en
columnas distintas (i1 != i2), las matrices simplemente superponen
ambas acciones. Si i1 == i2 (mismo fuente, dos destinos), la
poblacion se reparte segun sin^2(theta_k) para cada componente,
donde theta_k = Omega_k * t. Para un pulso con duracion fijada por
la transicion mas lenta (Sec. SC), el angulo de Rabi de cada
componente difiere, y la fraccion que va a k=1 es sin^2(Omega_k*t).
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Dynamics.py vive en physics/ → imports relativos al mismo paquete
try:
    from physics.hamiltonian_cah import CaHPlusHamiltonian, MolecularConstants
    from physics.pulses_cah import PulseLibrary, Pulse
except ImportError:
    from hamiltonian_cah import CaHPlusHamiltonian, MolecularConstants
    from pulses_cah import PulseLibrary, Pulse


# =====================================================================
# IdealTransitionSolver
# =====================================================================

class IdealTransitionSolver:
    """
    Matrices de transicion analiticas para pulsos pi BSB ideales.

    Uso rapido (compatible con notebook Step 02):
        solver   = IdealTransitionSolver(ham, lib)
        matrices = solver.build()          # lista de T_a = A0+A1 (NS x NS)
        p_new    = solver.apply(action, p) # pre-medicion

    Uso completo para el MDP (Ec. 4a-4b):
        p0, p1, P0, P1 = solver.step(action, P)
        # luego muestrea k segun p0, p1 y usa P0 o P1
    """

    def __init__(self, ham: CaHPlusHamiltonian, lib: PulseLibrary):
        self.ham = ham
        self.lib = lib
        self.NS  = ham.n_states

        self.A0_list: List[np.ndarray] = []
        self.A1_list: List[np.ndarray] = []
        self._built = False

    # ------------------------------------------------------------------
    # Construccion de matrices (un pulso)
    # ------------------------------------------------------------------

    def _build_one(self, pulse: Pulse) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula A0 y A1 para un pulso dado.

        Caso 1 — transicion simple (un solo par src->tgt):
            A0[:, src] = 0
            A1[tgt, src] = 1
            resto de la identidad sin cambio

        Caso 2 — multi-transicion (mismo src, dos tgt):
            Se modela con angulos de Rabi reales.
            Duracion del pulso D = pi / (lambda_LD * Omega_lenta)
            Para componente k con Omega_k:
                theta_k = Omega_k * lambda_LD * D
                fraccion en k=1: sin^2(theta_k)
                fraccion en k=0: cos^2(theta_k)
            Suma total de fracciones puede ser > 1 si las dos
            transiciones compiten en la misma columna, por lo que
            normalizamos para garantizar estocacidad.

        Caso 3 — multi-transicion (src distintos):
            Cada par actua en su propia columna; no hay competencia.
        """
        NS  = self.NS
        A0  = np.eye(NS, dtype=float)
        A1  = np.zeros((NS, NS), dtype=float)

        # Agrupar por src_eig
        src_map = {}
        for tr in pulse.transitions:
            src_map.setdefault(tr.src_eig, []).append(tr)

        LAMBDA_LD = 0.09
        D_s = pulse.duration_ms * 1e-3   # duracion en segundos

        for src, trans_list in src_map.items():

            if len(trans_list) == 1:
                # ------ Caso 1: pi-pulso perfecto ------
                tgt = trans_list[0].tgt_eig
                A0[src, src] = 0.0
                A1[tgt, src] = 1.0

            else:
                # ------ Caso 2: mismo src, dos destinos ------
                # Angulo de Rabi real para cada componente
                thetas = np.array([
                    tr.rabi_freq * LAMBDA_LD * D_s
                    for tr in trans_list
                ])
                sin2 = np.sin(thetas) ** 2
                cos2 = np.cos(thetas) ** 2

                # Fraccion que permanece en k=0:
                # producto de los cos^2 (ambas transiciones deben
                # dejar al estado en k=0 simultaneamente)
                frac_k0 = float(np.prod(cos2))

                # Fraccion que va a k=1 por cada componente:
                # sin^2 de esa componente
                fracs_k1 = sin2

                # Normalizar para garantizar estocacidad
                total = frac_k0 + fracs_k1.sum()
                if total > 1e-12:
                    frac_k0   = frac_k0   / total
                    fracs_k1  = fracs_k1  / total

                A0[src, src] = frac_k0
                for tr, fk1 in zip(trans_list, fracs_k1):
                    A1[tr.tgt_eig, src] += fk1

        return A0, A1

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def build(self) -> List[np.ndarray]:
        """
        Construye matrices para todos los pulsos.

        Returns
        -------
        matrices : list[ndarray (NS x NS)]
            T_a = A0 + A1 para cada pulso a.
            Matrices estocásticas: columnas suman 1.
        """
        self.A0_list = []
        self.A1_list = []

        for pulse in self.lib.pulses:
            A0, A1 = self._build_one(pulse)
            self.A0_list.append(A0)
            self.A1_list.append(A1)

        self._built = True
        return [A0 + A1 for A0, A1 in zip(self.A0_list, self.A1_list)]

    def _ensure_built(self):
        if not self._built:
            self.build()

    def apply(self, action: int, P: np.ndarray) -> np.ndarray:
        """
        Aplica el pulso SIN medicion: devuelve (A0+A1) @ P.

        Compatible con Step 02 del notebook:
            p_new = solver.apply(action, p)

        Nota: este es el vector pre-medicion. Para el MDP real
        usar step() que devuelve p0, p1 y los vectores post-medicion.
        """
        self._ensure_built()
        return (self.A0_list[action] + self.A1_list[action]) @ P

    def step(self, action: int,
             P: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Aplica el pulso y devuelve las dos ramas de medicion.

        Implementa Ec. 4a-4b del paper:
            p(k|S,a) = ||A^(a)_k S||_1
            S_{t+1} = A^(a)_k S_t  (normalizado)

        Parameters
        ----------
        action : int
        P      : ndarray (NS,)  vector de poblacion normalizado

        Returns
        -------
        p0     : float  probabilidad de medir k=0
        p1     : float  probabilidad de medir k=1
        P_post0: ndarray (NS,) post-medicion k=0, normalizado
        P_post1: ndarray (NS,) post-medicion k=1, normalizado
        """
        self._ensure_built()
        A0, A1 = self.A0_list[action], self.A1_list[action]
        v0 = A0 @ P
        v1 = A1 @ P
        p0 = float(v0.sum())
        p1 = float(v1.sum())
        P_post0 = v0 / p0 if p0 > 1e-12 else np.zeros(self.NS)
        P_post1 = v1 / p1 if p1 > 1e-12 else np.zeros(self.NS)
        return p0, p1, P_post0, P_post1

    def validate(self, tol: float = 1e-9) -> bool:
        """
        Verifica estocacidad y no-negatividad de todas las matrices.

        Returns True si todas son validas.
        """
        self._ensure_built()
        all_ok = True
        for i, (A0, A1) in enumerate(zip(self.A0_list, self.A1_list)):
            T = A0 + A1
            col_sums = T.sum(axis=0)
            neg = (T < -tol).any()
            bad_sum = not np.allclose(col_sums, 1.0, atol=tol)
            if neg or bad_sum:
                pid = self.lib.pulses[i].table_id or i
                print(f"  Pulso {pid}: col_sum=[{col_sums.min():.6f},"
                      f"{col_sums.max():.6f}]  neg={neg}")
                all_ok = False
        return all_ok


# =====================================================================
# ProjectiveMeasurement
# =====================================================================

class ProjectiveMeasurement:
    """
    Medicion proyectiva del estado motional (Ec. S2-S3).

    La medicion colapsa el sistema segun el resultado del ion logico:
      - k=0 (no fluoresce): el sistema esta en la rama A0
      - k=1 (fluoresce):    el sistema esta en la rama A1

    Compatible con Step 02 del notebook:
        k, p_new = meas.sample(p_pre_medicion)

    NOTA: para el uso correcto en el MDP, primero llama
    solver.step(action, P) para obtener p0, p1, P_post0, P_post1,
    y luego muestrea k segun [p0, p1]. El metodo sample() aqui
    implementa eso de forma integrada dado P pre-medicion.

    Para el MDP del paper, el flujo correcto es:
        p0, p1, P0, P1 = solver.step(action, P)
        k = rng.choice([0,1], p=[p0, p1])
        P_next = P0 if k==0 else P1
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, P: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Muestrea un estado puro de la distribucion P.

        Usado en Step 02 despues de solver.apply() (pre-medicion).
        Interpreta P como distribucion sobre estados moleculares
        post-pulso-post-cooling y colapsa estocasticamente.

        Returns
        -------
        k     : int       indice del estado colapsado
        P_new : ndarray   vector delta_{j,k} (estado puro)
        """
        P_norm = np.asarray(P, dtype=float)
        total  = P_norm.sum()
        if total < 1e-12:
            raise ValueError("Vector de poblacion nulo.")
        P_norm = P_norm / total

        # Corregir errores numericos pequeños
        P_norm = np.clip(P_norm, 0.0, None)
        P_norm /= P_norm.sum()

        k     = int(self.rng.choice(len(P_norm), p=P_norm))
        P_new = np.zeros(len(P_norm), dtype=float)
        P_new[k] = 1.0
        return k, P_new

    def sample_from_branches(self,
                             p0: float, p1: float,
                             P_post0: np.ndarray,
                             P_post1: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Medicion correcta segun el paper (Ec. S2-S3).

        Usa las ramas A0, A1 ya calculadas por solver.step().

        Returns
        -------
        k      : int      resultado (0 o 1)
        P_next : ndarray  vector post-medicion normalizado
        """
        probs = np.array([p0, p1])
        probs = np.clip(probs, 0.0, None)
        probs /= probs.sum()
        k = int(self.rng.choice(2, p=probs))
        return k, (P_post0 if k == 0 else P_post1)


# =====================================================================
# Test — reproduce Step 02 del notebook
# =====================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from physics.hamiltonian_cah import CaHPlusHamiltonian, MolecularConstants
    from physics.pulses_cah import PulseLibrary

    print("=" * 60)
    print("Step 02 — Pulse Library & Transition Matrices")
    print("=" * 60)

    # 2.1 Hamiltonian + pulse library
    ham = CaHPlusHamiltonian(MolecularConstants(J_max=2))
    ham.diagonalize()
    lib = PulseLibrary(ham, use_table_s2_only=False)
    print(f"\n{lib}")

    print("\nFirst 10 pulses:")
    for p in lib.pulses[:10]:
        tr  = p.transitions[0]
        src = ham.basis[tr.src_eig]
        tgt = ham.basis[tr.tgt_eig]
        print(f"  Pulse {p.index:3d}: "
              f"|J={src[0]},mJ={src[1]:+d},mI={src[2]:+.0f}> -> "
              f"|J={tgt[0]},mJ={tgt[1]:+d},mI={tgt[2]:+.0f}>  "
              f"dJ={p.delta_J:+d} dmJ={p.delta_mJ:+d}")

    # 2.2 Build ideal transition matrices
    solver   = IdealTransitionSolver(ham, lib)
    matrices = solver.build()
    print(f"\nBuilt {len(matrices)} transition matrices, "
          f"each shape {matrices[0].shape}")

    # 2.3 Verify stochasticity
    for i, T in enumerate(matrices):
        col_sums = T.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-9), \
            f"Matrix {i} not stochastic! min={col_sums.min():.8f}"
    print("All matrices are column-stochastic ✓")

    ok = solver.validate()
    if ok:
        print("All matrices non-negative and valid ✓")

    # 2.4 Manual pulse sequence
    rng  = np.random.default_rng(42)
    meas = ProjectiveMeasurement(rng)

    p = ham.boltzmann_population(300.0, use_eigenstates=True)
    print(f"\nInitial: most populated state = {np.argmax(p)}")

    for step in range(5):
        action = int(rng.integers(lib.n_actions))
        p      = solver.apply(action, p)
        k, p   = meas.sample(p)
        J, mJ, mI = ham.basis[k]
        print(f"Step {step+1}: pulse={action:3d} -> "
              f"measured |J={J}, mJ={mJ:+d}, mI={mI:+.0f}>")

    # 2.5 Verificacion MDP (Ec. 4a-4b)
    print("\nVerificacion MDP (Ec. 4a-4b):")
    P_th = ham.boltzmann_population(300.0, use_eigenstates=True)
    for action in range(min(3, lib.n_actions)):
        p0, p1, P0, P1 = solver.step(action, P_th)
        print(f"  Pulso {action}: p0={p0:.4f}  p1={p1:.4f}  "
              f"suma={p0+p1:.6f}  purity_k1={P1.max():.4f}")

    print("\nStep 02 completado correctamente.")