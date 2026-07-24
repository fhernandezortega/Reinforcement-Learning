"""
hamiltonian_cah.py — Hamiltoniano CaH+ (J=1,2) corregido y verificado
=======================================================================
Referencias:
  - Chou et al., Nature 545, 203 (2017)  [ref. 18 del RL-QLS], arXiv:1612.03926
  - Pipi, Tao, Wu, Narang, Leibrandt (RL-QLS), Ec. S9 / Sec. SC

Hamiltoniano rotacional-hiperfino (Ec. S9 / Chou Ec. 1), en unidades de
frecuencia (Hz):

    H/h = R J(J+1)
          - g   (muN/h) B  Jz
          - gI  (muN/h) B  Iz
          - cIJ [ Iz Jz + 1/2 (I+ J- + I- J+) ]

DOS JUEGOS DE CONSTANTES, con procedencia explicita:

  1) chou2017()  — FISICAS, Chou Tabla II (v=0, J=0..14), transcritas del PDF:
       g   = -1.35 (J=0..2) ... -1.31 (J=14)
       cIJ = 8.27 (J=0) ... 8.18 kHz (J=14);  8.26 kHz para J=1,2
     Usar para: fisica real del CaH+, tasas BBR, extensiones J<=4 / J<=6
     (Fig. S7), donde no hay tablas del RL-QLS que igualar.

  2) rlqls_effective() — EFECTIVAS, obtenidas por ajuste de minimos cuadrados
     de ESTE Hamiltoniano a las 16 frecuencias de la Tabla S2 del RL-QLS
     (NO provienen de Chou):
       g = -1.390,  cIJ = {1: 8.196, 2: 8.059} kHz,  B = 0.357 mT
     Reproducen la S2 a rms 2.5 Hz / max 4.7 Hz. Usar para: benchmarks
     exactos contra Fig. 2 / Fig. S2 / Tabla S2 / Tabla I.

  HALLAZGO DOCUMENTADO (jul 2026): la Ec. S9 con las constantes publicadas
  de Chou NO reproduce la Tabla S2 del RL-QLS bajo ninguna eleccion de B:
     Chou puro (B=0.357):            rms 135 Hz, max 305 Hz
     Chou, B libre (B->0.3616):      rms 103 Hz
     g=-1.35 fijo, (c1,c2,B) libres: rms  69 Hz
     todo libre -> (-1.390, 8.195, 8.059, B=0.3570): rms 2.6 Hz
  B queda anclado en 0.357 por el termino gI*muN*B del proton, asi que el
  g ~ -1.390 no es una degeneracion con el campo.

  RESUELTO (Chou PDF completo, repositorio NIST, jul 2026):
  (i) La "Tabla III" es la Extended Data Table 3: usa B=0.357 mT (de ahi
      nuestro B), pero sus valores TEORICOS incluyen el corrimiento por
      acoplamiento off-resonante a otros subniveles inducido por los haces
      Raman -> NO son autovalores desnudos; que el check de abajo de
      9.84/13.44 no coincida con 10.73/13.51 es lo esperado.
  (ii) Chou declara ±5% de incertidumbre teorica en g y cIJ (y hasta 13%
      de sensibilidad de base en g) -> el g_eff=-1.390 del RL-QLS (3% de
      -1.35) cae dentro de la banda declarada por la propia fuente.
      Hipotesis: RL-QLS tomo g/cIJ de la Extended Data Table 1 (valores en
      r0 por metodo/base) en vez de la Table 2 promediada vibracionalmente.
  (iii) El ancla Rabi experimental de Chou es 2.078(14) kHz (= Tabla S2,
      pulsos 10/11); el "2.087" del SM del RL-QLS es un typo por
      transposicion de digitos. calibrate() debe anclar a 2.078.

CONFIRMADO CONTRA CHOU (Ecs. 1 y 12-14): este codigo reproduce la solucion
cerrada de Chou a precision de maquina — Ec. 14 (energias) a ~1e-4 Hz sobre
una escala de 1e11 Hz, Ec. 12 (coeficientes, con Y=(Ea-Eb)/2) a 1e-16 — y la
regla xi es textual de Chou (signos relativos; en extremos, el signo de m).
La Ec. 1 NO tiene terminos extra: las discrepancias con la Tabla S2 (RL-QLS)
y con la Tabla III (Chou) son de NUMEROS/condiciones, no de fisica.

Otras constantes:
    R     = 144   GHz    (Chou Ec. 1, ref. Abe 2012; 4R = 0.576 THz — el
                          "0.57 THz" de la Fig. 2a del RL-QLS es truncado)
    gI    = 5.585695     (proton, CODATA; apantallamiento ~ppm, despreciable)
    B     = 0.357 mT     (Chou Tabla III; el texto del RL-QLS redondea a 0.36)
    muN/h = 7.622593 MHz/T (CODATA)

CORRECCIONES CLAVE respecto a versiones anteriores:
  1) Etiquetado de xi POR ESTRUCTURA DEL AUTOVECTOR (Chou Ec. 12), no por
     orden de energia. En la base del doblete
        (a = |mJ=m-1/2, mI=+1/2>,  b = |mJ=m+1/2, mI=-1/2>):
        - componentes con el MISMO signo  -> xi = +
        - componentes con signo OPUESTO   -> xi = -
     Con cIJ > 0 el acoplamiento fuera de la diagonal es negativo en todos
     los bloques, asi que xi=+ es SIEMPRE el estado inferior del doblete y
     xi=- el superior (consistente con Chou Ec. 14 y con la Fig. 2a).
  2) SIN calibracion a los 26.1/37.6 kHz de la Fig 2a: eso empuja cIJ hacia
     ~19.6 kHz (unfisico). Se usan constantes con procedencia documentada.

Ordenamiento I..XVI (identico a Fig. S5 del RL-QLS):
    dentro de cada J -> primero xi=+ (m ascendente), luego xi=- (m ascendente).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union

MUN_H = 7.622593e6            # muN/h  [Hz/T]
OMEGA_MOT = 2 * np.pi * 5.164e6   # [rad/s] modo motional OOP (NIST)

# ---------------------------------------------------------------------------
# Chou 2017, Tabla II (v=0): g y cIJ por manifold, J = 0..14  [FISICAS]
# ---------------------------------------------------------------------------
CHOU_G: Dict[int, float] = {
    0: -1.35, 1: -1.35, 2: -1.35, 3: -1.34, 4: -1.34,
    5: -1.34, 6: -1.34, 7: -1.34, 8: -1.33, 9: -1.33,
    10: -1.33, 11: -1.32, 12: -1.32, 13: -1.31, 14: -1.31,
}
CHOU_CIJ_HZ: Dict[int, float] = {J: c * 1e3 for J, c in {
    0: 8.27, 1: 8.26, 2: 8.26, 3: 8.26, 4: 8.26,
    5: 8.25, 6: 8.25, 7: 8.24, 8: 8.24, 9: 8.23,
    10: 8.22, 11: 8.21, 12: 8.20, 13: 8.19, 14: 8.18,
}.items()}

# Efectivas del RL-QLS [ajuste a Tabla S2; NO son de Chou]
RLQLS_G = -1.390
RLQLS_CIJ_HZ: Dict[int, float] = {1: 8.196e3, 2: 8.059e3}


@dataclass
class MolecularConstants:
    R:      float = 144.0e9     # Hz  (Chou Ec. 1, ref. Abe 2012)
    g:      Union[float, Dict[int, float]] = -1.35        # Chou T.II (J=1,2)
    gI:     float = 5.585695    # adimensional (proton, CODATA)
    cIJ:    Union[float, Dict[int, float]] = 8.26e3       # Hz, Chou T.II (J=1,2)
    B:      float = 0.357e-3    # T (Chou Tabla III)
    muN_h:  float = MUN_H       # Hz/T
    J_min:  int   = 1
    J_max:  int   = 2


def _collapse(d: Dict[int, float], Js) -> Union[float, Dict[int, float]]:
    vals = {J: d[J] for J in Js}
    return next(iter(vals.values())) if len(set(vals.values())) == 1 else vals


def chou2017(J_min: int = 1, J_max: int = 2) -> MolecularConstants:
    """Constantes FISICAS de Chou 2017 Tabla II (v=0) para J_min..J_max."""
    Js = range(J_min, J_max + 1)
    return MolecularConstants(g=_collapse(CHOU_G, Js),
                              cIJ=_collapse(CHOU_CIJ_HZ, Js),
                              J_min=J_min, J_max=J_max)


def rlqls_effective() -> MolecularConstants:
    """Constantes EFECTIVAS que reproducen la Tabla S2 del RL-QLS (J<=2).
    Provienen de un ajuste de minimos cuadrados, no de Chou. Solo para
    benchmarks contra las tablas/figuras del RL-QLS."""
    return MolecularConstants(g=RLQLS_G, cIJ=dict(RLQLS_CIJ_HZ),
                              J_min=1, J_max=2)


class CaHHamiltonian:

    def __init__(self, constants: Optional[MolecularConstants] = None):
        c = constants or MolecularConstants()
        self.R      = c.R
        self.g      = c.g
        self.gI     = c.gI
        self.cIJ    = c.cIJ
        self.B      = c.B
        self.muN_h  = c.muN_h
        self.J_manifolds = list(range(c.J_min, c.J_max + 1))
        self.omega_mot   = OMEGA_MOT

        # base producto |J, mJ, mI>
        self.prod_basis: List[Tuple[int, int, float]] = [
            (J, mJ, mI)
            for J in self.J_manifolds
            for mJ in range(-J, J + 1)
            for mI in (-0.5, 0.5)
        ]
        self.n_states = len(self.prod_basis)

        self._diagonalize()

    # ------------------------------------------------------------------
    # Constantes por manifold (o escalares)
    # ------------------------------------------------------------------
    def _per_J(self, val, J, name):
        if isinstance(val, dict):
            if J not in val:
                raise KeyError(f"{name} no definido para J={J}: añade el "
                               f"valor de Chou Tabla II")
            return val[J]
        return val

    def _c(self, J):
        return self._per_J(self.cIJ, J, "cIJ")

    def _g(self, J):
        return self._per_J(self.g, J, "g")

    # ------------------------------------------------------------------
    # Construccion de bloques (J, m) en Hz
    # ------------------------------------------------------------------
    def _block_states(self, J, m):
        # ordena de modo que st[0]=|mJ=m-1/2,mI=+1/2>=a, st[1]=|mJ=m+1/2,mI=-1/2>=b
        return [(mJ, mI)
                for mJ in range(-J, J + 1)
                for mI in (-0.5, 0.5)
                if abs(mJ + mI - m) < 1e-9]

    def _block_H(self, J, m):
        st = self._block_states(J, m)
        n = len(st)
        cJ = self._c(J)
        gJ = self._g(J)
        H = np.zeros((n, n))
        E_rot = self.R * J * (J + 1)
        for k, (mJ, mI) in enumerate(st):
            E_Z  = -gJ * mJ * self.muN_h * self.B \
                   - self.gI * mI * self.muN_h * self.B
            E_SR = -cJ * mJ * mI                       # diagonal Iz Jz
            H[k, k] = E_rot + E_Z + E_SR
        if n == 2:
            assert st[0][1] == +0.5 and st[1][1] == -0.5, \
                "orden del doblete (a,b) roto: revisa _block_states"
            (mJa, mIa), _ = st
            I = 0.5
            off = -cJ * 0.5 \
                  * np.sqrt(J * (J + 1) - mJa * (mJa + 1)) \
                  * np.sqrt(I * (I + 1) - mIa * (mIa - 1))
            H[0, 1] = H[1, 0] = off
        return H, st

    @staticmethod
    def _xi_from_vec(vec2):
        """xi por estructura de signo del autovector (Chou Ec. 12)."""
        ca, cb = vec2[0], vec2[1]
        return '+' if (ca * cb) > 0 else '-'

    def _mvalues(self, J):
        return sorted(set(mJ + mI
                          for mJ in range(-J, J + 1)
                          for mI in (-0.5, 0.5)))

    # ------------------------------------------------------------------
    # Diagonalizacion -> autobase |J, m, xi>, orden I..XVI
    # ------------------------------------------------------------------
    def _diagonalize(self):
        raw = []   # (J, m, xi, E_hz, vec_en_base_producto)
        for J in self.J_manifolds:
            for m in self._mvalues(J):
                Hb, st = self._block_H(J, m)
                ev, Vb = np.linalg.eigh(Hb)
                for k in range(len(ev)):
                    if len(ev) == 1:
                        xi = '+' if m > 0 else '-'       # extremo: signo de m
                    else:
                        xi = self._xi_from_vec(Vb[:, k])
                    vec = np.zeros(self.n_states)
                    for r, (mJ, mI) in enumerate(st):
                        vec[self.prod_basis.index((J, mJ, mI))] = Vb[r, k]
                    raw.append((J, m, xi, ev[k], vec))

        # orden I..XVI: (J, xi=+ antes que -, m ascendente)
        raw.sort(key=lambda x: (x[0], 0 if x[2] == '+' else 1, x[1]))

        self.eig_labels  = [(J, m, xi) for (J, m, xi, E, v) in raw]
        self.energies_hz = np.array([E for (J, m, xi, E, v) in raw])
        self.V           = np.array([v for (J, m, xi, E, v) in raw]).T  # NS x NS
        # adimensional (E/omega_mot); omega_mot/2pi = 5.164e6
        self.energies    = self.energies_hz / (self.omega_mot / (2 * np.pi))
        self.basis       = self.eig_labels   # alias de compatibilidad

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------
    def build(self):
        """H diagonal 16x16 en la autobase (adimensional, omega_mot=1)."""
        return np.diag(self.energies).astype(complex)

    def build_hz(self):
        """H diagonal 16x16 en la autobase (Hz)."""
        return np.diag(self.energies_hz).astype(complex)

    def get_transformation(self):
        """V: columna k = autoestado k (orden I..XVI) en base producto."""
        return self.V

    def diagonalize(self):
        """Compat: (energias_adimensionales, V)."""
        return self.energies.copy(), self.V.copy()

    def get_labels(self):
        roman = ['I','II','III','IV','V','VI','VII','VIII','IX','X',
                 'XI','XII','XIII','XIV','XV','XVI']
        return [f"{roman[i]}: |J={J}, m={m:+.1f}, xi={xi}>"
                for i, (J, m, xi) in enumerate(self.eig_labels)]

    def get_boltzmann(self, T=300.0, dtype=np.float64):
        """Boltzmann sobre los 16 autoestados (incluye offset rotacional R J(J+1))."""
        kB = 1.380649e-23
        h  = 6.62607015e-34
        E_J = self.energies_hz * h
        beta = 1.0 / (kB * T)
        w = np.exp(-beta * (E_J - E_J.min()))
        return (w / w.sum()).astype(dtype)

    def boltzmann_population(self, T=300.0, **kw):
        return self.get_boltzmann(T)

    def state_index(self, J, m, xi):
        return self.eig_labels.index((J, m, xi))

    # ------------------------------------------------------------------
    # Verificaciones
    # ------------------------------------------------------------------
    def check_table_III(self, verbose=True):
        """INFORMATIVO (no pass/fail): lanzaderas |J,-J+1/2,-> <-> |J,-J-1/2,->
        vs Extended Data Table 3 de Chou.

        Los valores de esa tabla estan a B=0.357 mT PERO incluyen el
        corrimiento por acoplamiento off-resonante a otros subniveles
        inducido por los haces Raman (lo dice su caption), mas ±5% de
        incertidumbre teorica en g y cIJ. Por eso el autovalor desnudo
        (9.84 / 13.44 con constantes de Chou) queda por debajo de los
        10.94/10.73 y 13.55/13.51 tabulados: es lo esperado, no un error.
        El benchmark desnudo correcto es check_table_S2.
        """
        ref = {1: (10.94, 10.73), 2: (13.55, 13.51)}
        out = {}
        for J in self.J_manifolds:
            i_T   = self.state_index(J, -J + 0.5, '-')
            i_fin = self.state_index(J, -J - 0.5, '-')
            f = abs(self.energies_hz[i_T] - self.energies_hz[i_fin]) / 1e3
            out[J] = f
            if verbose and J in ref:
                exp, teo = ref[J]
                print(f"  J={J}:  calc = {f:6.2f} kHz "
                      f"(ED-T3 exp {exp}, teo {teo}; con corrimiento de drive)")
        return out

    def check_table_S2(self, verbose=True):
        """Las 16 frecuencias de transicion de la Tabla S2 del RL-QLS (kHz).
        Con rlqls_effective(): rms ~2.5 Hz. Con chou2017(): rms ~135 Hz
        (inconsistencia documentada del RL-QLS con su fuente)."""
        S2 = [  # (pulso, estado_i, estado_f, f_S2_kHz)
            (1, (2, +2.5, '+'), (2, +1.5, '+'),  -1.72),
            (2, (2, +1.5, '+'), (2, +0.5, '+'),  -1.44),
            (3, (1, +1.5, '+'), (1, +0.5, '+'),  -1.06),
            (3, (2, +0.5, '+'), (2, -0.5, '+'),  -1.01),
            (4, (1, +0.5, '+'), (1, -0.5, '+'),  -0.30),
            (4, (2, -0.5, '+'), (2, -1.5, '+'),  -0.17),
            (5, (2, -1.5, '+'), (2, -2.5, '-'),   4.40),
            (6, (1, -0.5, '+'), (1, +0.5, '-'),  26.13),
            (7, (2, +1.5, '-'), (2, +0.5, '-'),  -6.12),
            (8, (2, +0.5, '-'), (2, -0.5, '-'),  -6.56),
            (9, (1, +0.5, '-'), (1, -0.5, '-'),  -7.26),
            (9, (2, -0.5, '-'), (2, -1.5, '-'),  -7.40),
            (11, (1, -0.5, '-'), (1, -1.5, '-'), -9.87),
            (13, (2, -1.5, '-'), (2, -2.5, '-'), -13.13),
        ]
        errs = []
        for n, a, b, fr in S2:
            fm = (self.energies_hz[self.state_index(*b)]
                  - self.energies_hz[self.state_index(*a)]) / 1e3
            errs.append(fm - fr)
            if verbose:
                print(f"  P{n:2d}: {fm:8.3f} vs {fr:7.2f}  Δ={fm-fr:+6.3f} kHz")
        errs = np.array(errs)
        if verbose:
            print(f"  rms = {np.sqrt((errs**2).mean())*1e3:.1f} Hz | "
                  f"max = {np.abs(errs).max()*1e3:.1f} Hz")
        return errs


CaHPlusHamiltonian = CaHHamiltonian


if __name__ == "__main__":
    print("=" * 60)
    print("CaH+  Hamiltoniano — validacion con ambos presets")
    print("=" * 60)

    for tag, mc in [("Chou 2017 (fisicas)  ", chou2017()),
                    ("RL-QLS (efectivas)   ", rlqls_effective())]:
        ham = CaHHamiltonian(mc)
        e = ham.check_table_S2(verbose=False)
        print(f"{tag}: Tabla S2 rms={np.sqrt((e*e).mean())*1e3:6.1f} Hz  "
              f"max={np.abs(e).max()*1e3:6.1f} Hz")

    print("\nDetalle Tabla S2 con rlqls_effective():")
    ham_eff = CaHHamiltonian(rlqls_effective())
    ham_eff.check_table_S2()

    print("\nAutoestados (orden I..XVI, preset Chou):")
    ham_chou = CaHHamiltonian(chou2017())
    for lab in ham_chou.get_labels():
        print("  ", lab)

    print("\nTabla III de Chou (preset Chou):")
    ham_chou.check_table_III()

    b = ham_chou.get_boltzmann(300.0)
    print(f"\nBoltzmann 300K: sum={b.sum():.6f}, purity(max)={b.max():.4f}")
    print(f"V unitaria? ||V V^T - I|| = "
          f"{np.linalg.norm(ham_chou.V @ ham_chou.V.T - np.eye(ham_chou.n_states)):.2e}")