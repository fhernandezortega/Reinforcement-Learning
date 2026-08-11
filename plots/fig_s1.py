"""
fig_S1.py — Oscilaciones de Rabi |J,-J-1/2,-> <-> |J,-J+1/2,-> para J=1,2 (Fig. S1).
Portadora (dos fotones Raman) del SISTEMA COMPLETO (16 niveles) con mesolve, en el
marco rotante resonante con la transicion ancla. Estado inicial: |J,-J-1/2,->
(singlete extremo inferior, estado producto puro, xi=-). Sin ruido.
Las ondulaciones finas vienen del acoplamiento off-resonante al resto de estados
(anarmonicidad de la escalera); no aparecen en un modelo de 2 niveles aislado.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qutip as qt
from physics.hamiltonian_cah import CaHHamiltonian, chou2017
from physics.Raman_rates import RamanRates

ham = CaHHamiltonian(chou2017())
L  = ham.eig_labels
E  = ham.energies_hz
NS = ham.n_states
M  = np.array([m for (_, m, _) in L])          # m de cada autoestado
rates = RamanRates(ham); rates.calibrate()


def all_trans(dm):
    out = []
    for i, (Ji, mi, _) in enumerate(L):
        for f, (Jf, mf, _) in enumerate(L):
            if Ji == Jf and abs((mf - mi) - dm) < 1e-9:
                Om = max(rates.omega_hz(i, f), rates.omega_hz(f, i))
                if Om >= 50:
                    out.append((i, f, Om))
    return out


def simulate(J, t_ms):
    """Rabi de la ancla |J,-J-1/2,-> -> |J,-J+1/2,-> (Delta m=+1), marco rotante
    resonante con ella."""
    m0    = -J - 0.5                          # extremo inferior (producto puro, xi=-)
    start = L.index((J, m0, '-'))            # IV (J=1) / XII (J=2)
    dest  = L.index((J, m0 + 1.0, '-'))      # V  (J=1) / XIII (J=2)

    tp = 2 * np.pi
    w  = tp * E
    wL = w[dest] - w[start]                   # frecuencia del drive (resonante)
    # d_j=(m_j-m0)*wL hace estaticos TODOS los acoples Delta m=+1; start y dest
    # quedan degenerados (resonantes) -> flop completo. El resto queda detuneado
    # por la anarmonicidad de la escalera -> ondulaciones finas.
    diag = (w - w[start]) - (M - m0) * wL
    H0 = np.diag(diag)
    Hc = np.zeros((NS, NS), dtype=complex)
    for (i, f, Om_hz) in all_trans(+1):       # acoples de la portadora (Hc simetrico)
        g = tp * Om_hz / 2.0
        Hc[f, i] += g; Hc[i, f] += g
    H = qt.Qobj(H0 + Hc)

    psi0  = qt.basis(NS, start)
    e_ops = [qt.basis(NS, start) * qt.basis(NS, start).dag(),
             qt.basis(NS, dest)  * qt.basis(NS, dest).dag()]
    res = qt.mesolve(H, psi0, t_ms * 1e-3, c_ops=[], e_ops=e_ops)
    return res.expect, (start, dest)


t_ms = np.linspace(0, 1.1, 800)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, J in zip(axes, (1, 2)):
    exps, (start, dest) = simulate(J, t_ms)
    ax.plot(t_ms, exps[0], color="C0", lw=1.3, label=f"|{J},{-J-0.5:+.1f},->")
    ax.plot(t_ms, exps[1], color="C1", lw=1.3, label=f"|{J},{-J+0.5:+.1f},->")
    ax.set_title(f"Rabi oscillations  J = {J}  (no damping)")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig("fig_S1.png", dpi=150, bbox_inches="tight")
print("Fig S1 generada (marco rotante, sistema completo 16 niveles)")
for J in (1, 2):
    lo = L.index((J, -J - 0.5, '-')); up = L.index((J, -J + 0.5, '-'))
    Om = max(rates.omega_hz(lo, up), rates.omega_hz(up, lo))
    print(f"  J={J}: |{J},{-J-0.5:+.1f},-> -> |{J},{-J+0.5:+.1f},->  "
          f"Omega={Om/1e3:.4f} kHz, periodo={1e6/Om:.0f} us")