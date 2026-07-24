"""
pulses_cah.py — Biblioteca de pulsos Raman CaH+ (Tabla S2) sobre QuTiP.
======================================================================
Adaptado para el nuevo Hamiltoniano en AUTOBASE |J,m,xi> (I..XVI).

- Las tasas Omega, frecuencias f y duraciones D vienen DIRECTO de la Tabla S2
  (no se recalculan).
- Los estados se expresan en la autobase (J, m, xi), no en base producto.
- Regla fisica: ΔJ=0, Δm=±1 (Chou: los pulsos no cambian J).
- Evolucion Schrodinger con QuTiP en el espacio mol x {k=0,1}, frame rotante,
  H0 completo -> detunings reales de todos los estados (fuga off-resonante).
- Salida: matrices A0, A1 (Ec. 4b), columna-estocasticas.

Unidades internas: rad/s y s. No depende de physics.units.
"""
import numpy as np
import qutip as qt

try:
    from physics.hamiltonian_cah import CaHHamiltonian      # repo
except ModuleNotFoundError:
    from hamiltonian_cah import CaHHamiltonian              # local

LAMBDA_LD = 0.09
NU_MOT_HZ = 5.164e6

# ============================================================
# TABLA S2  (tal cual del paper) + mapeo a autoestados (J,m,xi)
#   f_kHz : frecuencia de la transicion (columna f de la Tabla S2)
#   D_2pi : duracion en unidades 2pi^-1 ms  (D_real_ms = D_2pi / 2pi)
#   trans : lista de (estado_A, estado_B, Omega_kHz)
#           el par es no-ordenado; la direccion i->f se fija por signo de f
# ============================================================
PULSE_TABLE = [
    dict(id=1,  f_kHz=-1.72, D_2pi=16.2,
         trans=[((2, 2.5,'+'), (2, 1.5,'+'), 2.156)]),
    dict(id=2,  f_kHz=-1.44, D_2pi=34.6,
         trans=[((2, 1.5,'+'), (2, 0.5,'+'), 1.008)]),
    dict(id=3,  f_kHz=-1.03, D_2pi=52.6,
         trans=[((1, 1.5,'+'), (1, 0.5,'+'), 2.138),
                ((2, 0.5,'+'), (2,-0.5,'+'), 0.621)]),
    dict(id=4,  f_kHz=-0.23, D_2pi=18.7,
         trans=[((1, 0.5,'+'), (1,-0.5,'+'), 1.857),
                ((2,-0.5,'+'), (2,-1.5,'+'), 1.881)]),
    dict(id=5,  f_kHz= 4.40, D_2pi=28.5,
         trans=[((2,-1.5,'+'), (2,-2.5,'-'), 1.223)]),
    dict(id=6,  f_kHz=26.13, D_2pi=29.7,
         trans=[((1, 0.5,'-'), (1,-0.5,'+'), 1.174)]),
    dict(id=7,  f_kHz=-6.12, D_2pi=16.6,
         trans=[((2, 1.5,'-'), (2, 0.5,'-'), 2.097)]),
    dict(id=8,  f_kHz=-6.56, D_2pi=56.2,
         trans=[((2, 0.5,'-'), (2,-0.5,'-'), 0.621)]),
    dict(id=9,  f_kHz=-7.33, D_2pi=23.7,
         trans=[((1, 0.5,'-'), (1,-0.5,'-'), 1.857),
                ((2,-0.5,'-'), (2,-1.5,'-'), 1.221)]),
    dict(id=10, f_kHz= 9.87, D_2pi=16.8,
         trans=[((1,-1.5,'-'), (1,-0.5,'-'), 2.078)]),
    dict(id=11, f_kHz=-9.87, D_2pi=16.8,
         trans=[((1,-1.5,'-'), (1,-0.5,'-'), 2.078)]),
    dict(id=12, f_kHz=13.13, D_2pi=18.8,
         trans=[((2,-2.5,'-'), (2,-1.5,'-'), 1.852)]),
    dict(id=13, f_kHz=-13.13, D_2pi=18.8,
         trans=[((2,-2.5,'-'), (2,-1.5,'-'), 1.852)]),
]


class CaHPulses:

    LAMBDA_LD = LAMBDA_LD

    def __init__(self, ham=None):
        self.ham    = ham or CaHHamiltonian()
        self.N      = self.ham.n_states
        self.E_hz   = self.ham.energies_hz          # Hz, autobase I..XVI
        self.labels = self.ham.eig_labels           # [(J,m,xi), ...]
        self.nu_mot = NU_MOT_HZ
        self.pulse_library = self._build_pulse_library()

    # ---- utilidades ----
    def _idx(self, label):
        J, m, xi = label
        return self.labels.index((J, float(m), xi))

    def _order_if(self, A, B, f_kHz):
        """Devuelve (i,f) tal que E_f - E_i tenga el signo de f_kHz (direccion BSB)."""
        ia, ib = self._idx(A), self._idx(B)
        dE = (self.E_hz[ib] - self.E_hz[ia]) / 1e3   # kHz
        return (ia, ib) if np.sign(dE) == np.sign(f_kHz) else (ib, ia)

    def _build_pulse_library(self):
        lib = []
        for p in PULSE_TABLE:
            D_s = p["D_2pi"] * 1e-3 / (2 * np.pi)         # s (Tabla S2)
            couplings = []
            dE_list = []
            for (A, B, Om_kHz) in p["trans"]:
                i, f = self._order_if(A, B, p["f_kHz"])
                couplings.append((i, f, Om_kHz * 1e3))    # (i, f, Omega_Hz)
                dE_list.append(self.E_hz[f] - self.E_hz[i])
            # laser sintonizado a la energia REAL del Hamiltoniano:
            # BSB con frecuencia = promedio de las transiciones (nota Tabla S2)
            f_L = self.nu_mot + float(np.mean(dE_list))
            lib.append(dict(id=p["id"], f_L=f_L, D_s=D_s,
                            couplings=couplings, f_kHz=p["f_kHz"]))
        return lib

    # ---- motor QuTiP (frame rotante, mol x {0,1}) ----
    def _H_rot(self, couplings, f_L):
        N = self.N
        H = np.zeros((2 * N, 2 * N), dtype=complex)
        tp = 2 * np.pi
        E_ref = self.E_hz[couplings[0][0]]
        for j in range(N):
            H[j, j]         = tp * (self.E_hz[j] - E_ref)
            H[N + j, N + j] = tp * (self.E_hz[j] + self.nu_mot - f_L - E_ref)
        for (i, f, Om_hz) in couplings:
            g = tp * (self.LAMBDA_LD * Om_hz) / 2.0
            H[N + f, i] += g
            H[i, N + f] += np.conj(g)
        return H

    def transition_matrices(self, pulse):
        N = self.N
        U = (-1j * qt.Qobj(self._H_rot(pulse["couplings"], pulse["f_L"]))
             * pulse["D_s"]).expm().full()
        A0 = np.abs(U[:N, :N]) ** 2
        A1 = np.abs(U[N:, :N]) ** 2
        return A0, A1

    def precompute_matrices(self):
        A0_list, A1_list = [], []
        for p in self.pulse_library:
            A0, A1 = self.transition_matrices(p)
            A0_list.append(A0)
            A1_list.append(A1)
        return A0_list, A1_list


# ======================= TEST =======================
if __name__ == "__main__":
    P = CaHPulses()
    rom = ['I','II','III','IV','V','VI','VII','VIII','IX','X',
           'XI','XII','XIII','XIV','XV','XVI']
    def name(k): 
        J,m,xi = P.labels[k]; return f"{rom[k]}|{J},{m:+.1f},{xi}>"

    print(f"Biblioteca: {len(P.pulse_library)} pulsos\n")
    print(f"{'P':>2} {'f(kHz)':>7} {'D(ms)':>7}  transiciones i->f (Ω kHz) | p(k=1) primaria")
    for p in P.pulse_library:
        A0, A1 = P.transition_matrices(p)
        parts = []
        for (i, f, Om) in p["couplings"]:
            s = np.zeros(P.N); s[i] = 1.0
            p1 = float((A1 @ s).sum())
            parts.append(f"{name(i)}->{name(f)} (Ω={Om/1e3:.3f}) p1={p1:.3f}")
        # chequeo de conservacion sobre TODOS los estados iniciales
        colsum = (A0 + A1).sum(axis=0)
        ok = np.allclose(colsum, 1.0, atol=1e-6)
        print(f"{p['id']:>2} {p['f_kHz']:>7.2f} {p['D_s']*1e3:>7.3f}  "
              + " ; ".join(parts) + f"   [Σcol=1: {ok}]")

    A0L, A1L = P.precompute_matrices()
    print(f"\nprecompute_matrices -> {len(A0L)} A0, {len(A1L)} A1, "
          f"shape {A0L[0].shape}")


# alias de compatibilidad con el codigo anterior
CaHPulse = CaHPulses