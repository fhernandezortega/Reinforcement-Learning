
"""
tdse_solver.py — Matrices de transicion A0/A1 via TDSE (Ecs. S4-S5).
====================================================================
Consume directamente generate_pulses_cah (generate_nist_library /
generate_pulse_library), sin depender de pulses_cah.py.
"""
import os, sys, pickle, time
import numpy as np
import qutip as qt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics.hamiltonian_cah import CaHHamiltonian, rlqls_effective
from physics.generate_pulses_cah import (generate_nist_library,
                                         generate_pulse_library, LAMBDA_LD)
from physics.Raman_rates import RamanRates


NU_MOT_HZ = 5.164e6   # modo motional OOP (NIST)
OMEGA_MIN_HZ = 50.0   # corte para incluir una transicion en el Hamiltoniano

# D tabuladas (Tabla S2, 2pi^-1 ms) para pulsos que NO siguen la regla general.
# Pulso 3: el paper lo marca como excepcion ("except for pulse 3"); su
# transferencia es hipersensible a D (regimen sobre-rotado de la transicion
# rapida), asi que se fuerza el valor tabulado.
D_TABULADA_OVERRIDE = {3: 52.6}


def _pulse_to_solver(ham, p):
    """Convierte una entrada de la libreria (f_kHz, D_2pi, trans, paper_id)
    al formato interno del solver, aplicando overrides de D tabulada.

    f_L va al PROMEDIO de las frecuencias de las transiciones del pulso
    (Sec. SC: "pulse frequencies chosen as the average of the two transition
    frequencies"). Anclar a UNA transicion desintoniza las demas por su
    separacion completa (~0.05-0.14 kHz ~ lam*Om) y hunde su transferencia.
    """
    labels = ham.eig_labels
    E = ham.energies_hz
    cpls = [(labels.index(li), labels.index(lf), Om * 1e3)
            for (li, lf, Om) in p["trans"]]
    f_trans = [E[f] - E[i] for (i, f, _) in cpls]       # Hz, por transicion
    f_L = NU_MOT_HZ + float(np.mean(f_trans))           # laser al promedio
    i0, f0, _ = cpls[0]
    dm = int(round(labels[f0][1] - labels[i0][1]))      # +1 o -1
    pid = p.get("paper_id")
    D_2pi = D_TABULADA_OVERRIDE.get(pid, p["D_2pi"])
    D_s = D_2pi / (2 * np.pi * 1e3)
    return dict(id=pid if pid is not None else "?", f_L=f_L, D_s=D_s,
                dm=dm, couplings=cpls)


class TDSESolverCaH:
    def __init__(self, ham=None, library=None, n_mot=4, rates=None,
                 cache_path=None, verbose=True):
        """
        ham      : CaHHamiltonian (default: preset efectivo RL-QLS, para
                   reproducir Fig. 2 / Tabla S2). Usar chou2017() para fisica.
        library  : lista de pulsos de generate_*; default generate_nist_library.
        n_mot    : truncamiento de Fock. 4 (como el paper); con 2 desaparece
                   la fuga |f,1> -> |siguiente,2> y todo pulso pi da 1.000.
        rates    : RamanRates ya calibrado (se crea si no se pasa).
        """
        self.ham = ham or CaHHamiltonian()  # default = Chou 2017 (fisicas)
        raw_lib = library if library is not None else \
            generate_nist_library(ham=self.ham)
        self.pulses = [_pulse_to_solver(self.ham, p) for p in raw_lib]

        self.NS = self.ham.n_states
        self.E_hz = self.ham.energies_hz
        self.n_mot = n_mot
        self.lam = LAMBDA_LD
        self.nu_mot = NU_MOT_HZ
        self.cache_path = cache_path
        self.verbose = verbose

        # todas las transiciones intra-J, agrupadas por dm (Ec. S4)
        self.rates = rates or RamanRates(self.ham)
        self.rates.calibrate()
        self._trans_by_dm = {-1: self._all_transitions(-1),
                             +1: self._all_transitions(+1)}

        a = qt.destroy(n_mot)
        self._a, self._ad, self._num = a, a.dag(), a.dag() * a
        self.A0_list, self.A1_list = [], []

    def _all_transitions(self, dm):
        """[(i, f, Omega_Hz)] de todas las transiciones intra-J con ese dm."""
        L = self.ham.eig_labels
        out = []
        for i, (Ji, mi, _) in enumerate(L):
            for f, (Jf, mf, _) in enumerate(L):
                if Ji == Jf and abs((mf - mi) - dm) < 1e-9:
                    Om = max(self.rates.omega_hz(i, f),
                             self.rates.omega_hz(f, i))
                    if Om >= OMEGA_MIN_HZ:
                        out.append((i, f, Om))
        return out

    # -------- Hamiltoniano rotante estatico (rad/s) --------
    def _H_rot(self, pulse):
        tp = 2 * np.pi
        w = tp * self.E_hz
        wL = tp * pulse["f_L"]
        wm = tp * self.nu_mot
        i0 = pulse["couplings"][0][0]
        wref = w[i0]

        H_mol = qt.Qobj(np.diag(w - wref))
        H = qt.tensor(H_mol, qt.qeye(self.n_mot)) \
            + qt.tensor(qt.qeye(self.NS), (wm - wL) * self._num)

        # TODAS las transiciones del dm del pulso (no solo las asignadas)
        for (i, f, Om_hz) in self._trans_by_dm[pulse["dm"]]:
            Om = tp * Om_hz
            op = qt.basis(self.NS, f) * qt.basis(self.NS, i).dag()
            bsb = (qt.tensor(op, self._ad) - qt.tensor(op.dag(), self._a))
            H += (Om / 2.0) * 1j * self.lam * bsb
        return H

    def _compute_one(self, pulse):
        H = self._H_rot(pulse)
        U = (-1j * H * pulse["D_s"]).expm().full()
        nm, NS = self.n_mot, self.NS
        A0 = np.zeros((NS, NS)); A1 = np.zeros((NS, NS))
        for l in range(NS):
            col = l * nm + 0                     # |l, n=0>
            for j in range(NS):
                A0[j, l] = abs(U[j * nm + 0, col]) ** 2
                A1[j, l] = sum(abs(U[j * nm + n, col]) ** 2
                               for n in range(1, nm))
        return A0, A1

    def compute(self):
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as fh:
                d = pickle.load(fh)
            self.A0_list, self.A1_list = d["A0_list"], d["A1_list"]
            if self.verbose:
                print(f"Cache cargada: {self.cache_path}")
            return self.A0_list, self.A1_list

        if self.verbose:
            n_tr = {k: len(v) for k, v in self._trans_by_dm.items()}
            print(f"TDSE (frame rotante, n_mot={self.n_mot}): "
                  f"{len(self.pulses)} pulsos, {self.NS} estados, "
                  f"transiciones por dm: {n_tr}\n")
        t0 = time.time()
        for idx, p in enumerate(self.pulses):
            A0, A1 = self._compute_one(p)
            self.A0_list.append(A0); self.A1_list.append(A1)
            if self.verbose:
                cs = (A0 + A1).sum(0)
                print(f"  [{idx + 1:2d}/{len(self.pulses)}] pulso {p['id']:>2}  "
                      f"Scol en [{cs.min():.5f},{cs.max():.5f}]")
        if self.verbose:
            print(f"\nTiempo total: {time.time() - t0:.2f} s")
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "wb") as fh:
                pickle.dump({"A0_list": self.A0_list,
                             "A1_list": self.A1_list}, fh)
        return self.A0_list, self.A1_list

    # -------- verificaciones --------
    def check_conservation(self):
        """A0+A1 debe conservar poblacion por columna (=1)."""
        worst = 0.0
        for A0, A1 in zip(self.A0_list, self.A1_list):
            cs = (A0 + A1).sum(0)
            worst = max(worst, np.abs(cs - 1).max())
        print(f"Conservacion (max |Scol - 1|): {worst:.2e}")
        return worst

    def transfers(self, pulse_id):
        """Transferencia P(|f>, n>=1) desde |i,0> de CADA transicion del pulso."""
        k = next(i for i, p in enumerate(self.pulses) if p["id"] == pulse_id)
        p = self.pulses[k]
        return {(i, f): self.A1_list[k][f, i] for (i, f, _) in p["couplings"]}

    # transferencias principales de la Fig. S2 (leidas de la figura)
    FIG_S2 = {1: 0.89, 2: 0.99, 3: 0.94, 4: 0.85, 5: 1.00, 6: 1.00, 7: 0.99,
              8: 0.97, 9: 0.74, 10: 1.00, 11: 1.00, 12: 1.00, 13: 1.00}

    def check_fig_S2(self, verbose=True):
        """Compara la transferencia principal de cada pulso con la Fig. S2."""
        errs = {}
        if verbose:
            print(f"{'P':>2} {'calc':>7} {'Fig.S2':>7} {'dif':>7}")
        for k, p in enumerate(self.pulses):
            if p["id"] not in self.FIG_S2:
                continue
            # La Fig. S2 rotula la transicion PRINCIPAL = la de mayor
            # transferencia. En multi-transicion asimetricos NO es la de mayor
            # Omega: la rapida se sobre-rota y pierde poblacion, mientras la
            # lenta completa mejor su pi (P3: lenta 0.943 vs Fig.S2 0.94;
            # P9: lenta 0.735 vs 0.74).
            val = max(self.A1_list[k][f, i] for (i, f, _) in p["couplings"])
            ref = self.FIG_S2[p["id"]]
            errs[p["id"]] = val - ref
            if verbose:
                print(f"{p['id']:>2} {val:7.3f} {ref:7.2f} {val-ref:+7.3f}")
        if verbose:
            e = np.array(list(errs.values()))
            print(f"  error medio = {np.abs(e).mean():.4f} | "
                  f"pulsos dentro de +-0.01: {int((np.abs(e) <= 0.01).sum())}/{len(e)}")
        return errs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mot", type=int, default=4)
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--preset", choices=["rlqls", "chou"], default="chou")
    args = ap.parse_args()

    from physics.hamiltonian_cah import chou2017
    ham = CaHHamiltonian(chou2017() if args.preset == "chou"
                         else rlqls_effective())

    print("=" * 60 + "\nTDSE Solver CaH+ (frame rotante)\n" + "=" * 60)
    s = TDSESolverCaH(ham=ham, n_mot=args.n_mot, cache_path=args.cache)
    s.compute()
    s.check_conservation()

    print("\nTransferencia principal vs Fig. S2:")
    s.check_fig_S2()
    print("\nListo.")