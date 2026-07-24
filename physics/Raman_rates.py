"""
Calculador de tasas Raman-Rabi angulares para CaH+ (Chou 2017, Ecs. 17-22).
Evita la fisica electronica: el prefactor E (Ec.17) es un escalar comun que
se CALIBRA con la transicion de referencia (2pi x 2.078 kHz — valor
EXPERIMENTAL de Chou, Methods, = Tabla S2 pulsos 10/11; el "2.087" del SM
del RL-QLS es un typo por transposicion). La parte angular usa los
coeficientes del autovector (columnas de V) + 3j de Wigner.

Haces (Chou, Methods): un haz pi (k=0) y un haz sigma- (k=-1), 1051 nm CW.
La configuracion excita Delta m = -1; la direccion inversa (Delta m = +1,
via sigma+) tiene identica |Omega| (Ec. S10 / Tabla S2: pulsos 10/11 y
12/13 comparten Omega) y se obtiene evaluando omega_hz(j, i).
Regla ΔJ=0 total (dos fotones), intermedios J_int = J ± 1.

Uso:  python Raman_rates.py            (listado Δm=-1 + cociente J1/J2)
      python Raman_rates.py validate   (tabla vs S2, presets Chou y RL-QLS)
"""
import sys

import numpy as np
from math import factorial as fac

import os
import sys
# Añadimos la ruta de la carpeta principal al sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics.hamiltonian_cah import CaHHamiltonian, chou2017, rlqls_effective

RATIO = 1.0/2.3   # (wbar-w1)/(wbar+w1) para 1051 nm (Chou: ~1:2.3)

def wig3j(j1,j2,j3,m1,m2,m3):
    if m1+m2+m3!=0: return 0.0
    if not (abs(j1-j2)<=j3<=j1+j2): return 0.0
    for j,m in ((j1,m1),(j2,m2),(j3,m3)):
        if abs(m)>j: return 0.0
    t1=j1+j2-j3; t2=j1-j2+j3; t3=-j1+j2+j3
    if min(t1,t2,t3)<0: return 0.0
    pref=np.sqrt(fac(int(t1))*fac(int(t2))*fac(int(t3))/fac(int(j1+j2+j3+1)))
    pref*=np.sqrt(fac(int(j1+m1))*fac(int(j1-m1))*fac(int(j2+m2))*
                  fac(int(j2-m2))*fac(int(j3+m3))*fac(int(j3-m3)))
    s=0.0
    kmin=int(max(0,j2-j3-m1,j1-j3+m2))
    kmax=int(min(j1+j2-j3,j1-m1,j2+m2))
    for k in range(kmin,kmax+1):
        den=(fac(k)*fac(int(j1+j2-j3-k))*fac(int(j1-m1-k))*
             fac(int(j2+m2-k))*fac(int(j3-j2+m1+k))*fac(int(j3-j1-m2+k)))
        s+=(-1)**k/den
    return pref*s*(-1)**int(j1-j2-m3)

def d(J1,m1,k,J2,m2):
    """<J1,m1| r.e_k |J2,m2>  (Chou Ec.21)."""
    if abs(J1-J2)!=1: return 0.0
    if -m1+k+m2!=0: return 0.0
    return (np.sqrt(max(J1,J2))*(J1-J2)*(-1)**(k+J1-m1)
            *wig3j(J1,1,J2,-m1,k,m2))

class RamanRates:
    def __init__(self, ham=None):
        self.ham=ham or CaHHamiltonian()
        self.V=self.ham.V
        self.prod=self.ham.prod_basis
        self.labels=self.ham.eig_labels
        self.E=self.ham.energies_hz

    def coeff(self, s, J, m, mI):
        """c^(s)_{mI}: comp. del autoestado s en |J,mJ=m-mI,mI>."""
        mJ=m-mI
        key=(J,int(mJ),mI)
        if key not in self.prod: return 0.0
        return self.V[self.prod.index(key), s]

    def _two_photon(self, sa, sb, order):
        """order='minus': pi(a)->sigma-(b) ; 'plus': sigma-(a)->pi(b)."""
        Ja,ma,_=self.labels[sa]; Jb,mb,_=self.labels[sb]
        if Ja!=Jb: return 0.0
        amp=0.0
        for mI in (-0.5,0.5):
            ca=self.coeff(sa,Ja,ma,mI); cb=self.coeff(sb,Jb,mb,mI)
            if ca==0 or cb==0: continue
            mJa=ma-mI; mJb=mb-mI
            for Jint in (Ja-1,Ja+1):
                if Jint<0: continue
                if order=='minus':
                    # a --pi(k=0)--> int --sigma-(k=-1)--> b
                    a1=d(Jint,mJa,0,Ja,mJa)          # <int,mJa|pi|a,mJa>
                    a2=d(Jb,mJb,-1,Jint,mJa)         # <b,mJb|sigma-|int,mJa>
                else:
                    # a --sigma-(k=-1)--> int --pi(k=0)--> b
                    a1=d(Jint,mJa-1,-1,Ja,mJa)       # <int,mJa-1|sigma-|a,mJa>
                    a2=d(Jb,mJb,0,Jint,mJa-1)        # <b,mJb|pi|int,mJa-1>
                amp+=cb*ca*a2*a1
        return amp

    def omega_rel(self, sa, sb):
        """Omega relativa (sin calibrar) para la transicion sa->sb."""
        Sm=self._two_photon(sa,sb,'minus')
        Sp=self._two_photon(sa,sb,'plus')
        return Sm + RATIO*Sp

    def omega_hz(self, sa, sb):
        """|Omega| calibrada en Hz para sa->sb.
        Devuelve 0 si la configuracion pi+sigma- no excita esa direccion
        (Delta m != -1); la direccion inversa se obtiene con omega_hz(sb,sa),
        que tiene identica |Omega| fisica (ver docstring del modulo)."""
        if not hasattr(self, "scale"):
            self.calibrate()
        return self.scale*abs(self.omega_rel(sa,sb))

    def calibrate(self):
        """Fija la escala con |Omega(V->IV)| = 2.078 kHz.
        Ancla: valor EXPERIMENTAL de Chou 2017 (Methods), Omega_{J=1} =
        2pi x 2.078(14) kHz = Tabla S2 del RL-QLS, pulsos 10/11.
        (El 2.087 del texto del SM del RL-QLS es un typo por transposicion.)"""
        a=self.labels.index((1,-0.5,'-'))   # V
        b=self.labels.index((1,-1.5,'-'))   # IV
        ref=abs(self.omega_rel(a,b))
        self.scale=2.078e3/ref
        return self.scale


def validate_against_S2(constants=None, titulo=""):
    R=RamanRates(CaHHamiltonian(constants)); R.calibrate(); L=R.labels
    # transiciones Δm=-1 intra-J con Ω calculada
    trans=[]
    for sa in range(16):
        Ja,ma,xa=L[sa]
        for sb in range(16):
            Jb,mb,xb=L[sb]
            if Ja==Jb and abs((mb-ma)+1)<1e-9:
                f=(R.E[sb]-R.E[sa])/1e3
                Om=R.omega_hz(sa,sb)/1e3
                if Om>0.05:
                    trans.append((f,Om,f"|{Ja},{ma:+.1f},{xa}>→|{Jb},{mb:+.1f},{xb}>"))
    # Tabla S2: (pulso, f_kHz, [Ω_kHz...])
    S2=[(1,-1.72,[2.156]),(2,-1.44,[1.008]),(3,-1.03,[0.621,2.138]),
        (4,-0.23,[1.881,1.857]),(5,4.40,[1.223]),(6,26.13,[1.174]),
        (7,-6.12,[2.097]),(8,-6.56,[0.621]),(9,-7.33,[1.221,1.857]),
        (10,9.87,[2.078]),(11,-9.87,[2.078]),(12,13.13,[1.852]),(13,-13.13,[1.852])]
    print(f"{'='*74}\nVALIDACION vs Tabla S2 (Ω en 2π·kHz){titulo}\n{'='*74}")
    print(f"{'P':>2} {'Ω_S2':>16} {'Ω_calc':>16} {'err%':>14}")
    for (p,fS2,OmS2) in S2:
        cands=sorted(trans,key=lambda t:abs(abs(t[0])-abs(fS2)))[:len(OmS2)]
        omcalc=sorted([c[1] for c in cands],reverse=True)
        oms2=sorted(OmS2,reverse=True)
        errs=[f"{100*(oc/os-1):+.2f}" for oc,os in zip(omcalc,oms2)]
        print(f"{p:>2} {str(oms2):>16} {str([f'{o:.3f}' for o in omcalc]):>16}"
              f" {str(errs):>14}")
    a1=L.index((1,-0.5,'-')); b1=L.index((1,-1.5,'-'))
    a2=L.index((2,-1.5,'-')); b2=L.index((2,-2.5,'-'))
    r=R.omega_hz(a1,b1)/R.omega_hz(a2,b2)
    print(f"Ω(J=1)/Ω(J=2) = {r:.4f}   (Chou: teo 1.132, exp 1.152(11); S2: 1.122)")


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="validate":
        validate_against_S2(chou2017(),      "  — preset Chou (fisicas)")
        print()
        validate_against_S2(rlqls_effective(), "  — preset RL-QLS (efectivas)")
    else:
        R=RamanRates()
        R.calibrate()
        L=R.labels
        def ix(J,m,xi): return L.index((J,m,xi))

        print("Transiciones Δm=-1 intra-J:  f=ΔE (kHz), Ω calculada (kHz)")
        rows=[]
        for sa in range(16):
            Ja,ma,xa=L[sa]
            for sb in range(16):
                Jb,mb,xb=L[sb]
                if Ja!=Jb: continue
                if abs((mb-ma)-(-1))>1e-9: continue
                f=(R.E[sb]-R.E[sa])/1e3
                Om=R.omega_hz(sa,sb)/1e3
                if Om>0.05:
                    rows.append((abs(f),Om,f,f"|{Ja},{ma:+.1f},{xa}>->|{Jb},{mb:+.1f},{xb}>"))
        for absf,Om,f,lab in sorted(rows):
            print(f"  |f|={absf:6.2f}  Ω={Om:6.3f}   {lab}")

        aJ1=ix(1,-0.5,'-'); bJ1=ix(1,-1.5,'-')
        aJ2=ix(2,-1.5,'-'); bJ2=ix(2,-2.5,'-')
        r=R.omega_hz(aJ1,bJ1)/R.omega_hz(aJ2,bJ2)
        print(f"\n  Ω(J=1 target)/Ω(J=2 target) = {r:.4f}"
              f"   (Chou: teo 1.132, exp 1.152(11))")