import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap

from physics.hamiltonian_cah import CaHHamiltonian, rlqls_effective
from physics.generate_pulses_cah import generate_nist_library

ROMAN = ['I','II','III','IV','V','VI','VII','VIII','IX','X',
         'XI','XII','XIII','XIV','XV','XVI']

def datos_fig2a():
    ham = CaHHamiltonian(rlqls_effective())
    L, E = ham.eig_labels, ham.energies_hz
    niveles = [{"idx": i, "roman": ROMAN[i], "J": J, "m": m, "xi": x,
                "E_kHz": E[i]/1e3} for i, (J, m, x) in enumerate(L)]
    flechas = []
    for p in generate_nist_library(ham=ham):
        for (li, lf, Om) in p["trans"]:
            flechas.append({"pulso": p["paper_id"], "i": L.index(li),
                            "f": L.index(lf), "Omega_kHz": Om})
    anot = {}
    for J in (1, 2):
        idx = [i for i,(Jl,m,x) in enumerate(L) if Jl==J]
        anot[J] = (max(E[i] for i in idx) - min(E[i] for i in idx))/1e3
    anot["dJ_THz"] = (E[L.index((2,-1.5,'+'))] - E[L.index((1,-0.5,'+'))])/1e12
    return niveles, flechas, anot

niveles, flechas, anot = datos_fig2a()
BYIDX = {n["idx"]: n for n in niveles}

RED  = "#e22323"
BG2  = "#d7e3f4"; BG1 = "#e7eef9"
BARW = 0.34
GAP  = 9.0

def _mlabel(v):
    num = round(v*2); return f"{num//2}" if num%2==0 else f"{num}/2"

def alturas(nJ):
    orden = sorted(nJ, key=lambda n: n["E_kHz"])
    return {n["idx"]: r for r, n in enumerate(orden)}

def _brace(ax, x, y0, y1, w=0.18):
    ym=(y0+y1)/2
    v=[(x,y0),(x+w,y0),(x+w,ym-0.03),(x+2*w,ym),(x+w,ym+0.03),(x+w,y1),(x,y1)]
    c=[Path.MOVETO,Path.CURVE3,Path.CURVE3,Path.CURVE3,Path.CURVE3,Path.CURVE3,Path.CURVE3]
    ax.add_patch(PathPatch(Path(v,c), fc="none", ec="black", lw=1.2, clip_on=False, zorder=5))

def dibujar_panel(ax, J, nJ, fJ, y0, kHz_val, show_B=False):
    rank = alturas(nJ)
    pos = {n["idx"]: (n["m"], rank[n["idx"]]+y0) for n in nJ}
    ms = sorted(set(n["m"] for n in nJ)); m_min,m_max = ms[0],ms[-1]
    n = len(nJ); ytop = y0+(n-1)

    ax.add_patch(FancyBboxPatch((m_min-0.75, y0-0.95), (m_max-m_min)+1.5, (n-1)+1.9,
                 boxstyle="round,pad=0.02,rounding_size=0.25",
                 fc=(BG2 if J==2 else BG1), ec="none", zorder=0))

    for nd in nJ:
        x,y = pos[nd["idx"]]
        ax.plot([x-BARW, x+BARW], [y,y], color="black", lw=1.6,
                solid_capstyle="round", zorder=3)
        ax.text(x-BARW*0.5, y-0.44, nd["roman"], ha="center", va="center",
                fontsize=8.5, zorder=3)

    grupos = defaultdict(list)
    for fl in fJ:
        grupos[tuple(sorted((fl["i"], fl["f"])))].append(fl)

    def draw(p0, p1, num, perp, off_line, off_lab):
        x0,y0a=p0; x1,y1a=p1; sh=perp*off_line
        ax.annotate("", xy=(x1+sh[0],y1a+sh[1]), xytext=(x0+sh[0],y0a+sh[1]),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.15, mutation_scale=11,
                shrinkA=7, shrinkB=7, linestyle=(0,(3.5,2))), zorder=2)
        mid=np.array([(x0+x1)/2,(y0a+y1a)/2])+perp*off_lab
        ax.text(mid[0],mid[1],str(num), color=RED, fontsize=8.5, fontweight="bold",
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.03", fc="white", ec="none", alpha=0.75))

    for fl in fJ:
        i,f = fl["i"], fl["f"]
        lo,hi = sorted((i,f))
        dv=np.array([pos[hi][0]-pos[lo][0], pos[hi][1]-pos[lo][1]]); dv=dv/np.hypot(*dv)
        perp=np.array([-dv[1], dv[0]])          # canonico: independiente de la direccion
        par=grupos[tuple(sorted((i,f)))]
        if len(par)>1:
            up = i<f
            draw(pos[i],pos[f],fl["pulso"], perp, 0.13 if up else -0.13, 0.46 if up else -0.46)
        else:
            draw(pos[i],pos[f],fl["pulso"], perp, 0.0, 0.32)

    ax.text(m_min-0.6, ytop+0.6, f"{kHz_val:.1f} kHz", fontsize=11, va="center")
    if show_B:
        ax.text(m_max+0.55, ytop+0.85, "B = 0.36 mT", fontsize=11, ha="right",
                va="center", style="italic")

    ya = y0-1.3
    ax.plot([m_min-0.7, m_max+0.95], [ya,ya], color="black", lw=1.1)
    for m in ms:
        ax.text(m, ya-0.52, _mlabel(m), ha="center", va="center", fontsize=10)
    ax.text(m_max+1.2, ya, "m", fontsize=12, style="italic", va="center")

    xb=m_max+0.62
    plus=[pos[nd["idx"]][1] for nd in nJ if nd["xi"]=='+']
    minus=[pos[nd["idx"]][1] for nd in nJ if nd["xi"]=='-']
    _brace(ax,xb,min(plus)-0.3,max(plus)+0.3)
    _brace(ax,xb,min(minus)-0.3,max(minus)+0.3)
    ax.text(xb+0.62, np.mean(plus), "+", fontsize=15, va="center")
    ax.text(xb+0.62, np.mean(minus), "\u2212", fontsize=15, va="center")

# ---------- ensamblaje (todo en un eje) ----------
fig = plt.figure(figsize=(7.4, 7.6))
ax = fig.add_axes([0.02, 0.03, 0.80, 0.90]); ax.axis("off")

nJ2=[n for n in niveles if n["J"]==2]; nJ1=[n for n in niveles if n["J"]==1]
fJ2=[f for f in flechas if BYIDX[f["i"]]["J"]==2]
fJ1=[f for f in flechas if BYIDX[f["i"]]["J"]==1]
dibujar_panel(ax, 2, nJ2, fJ2, GAP, anot[2], show_B=True)
dibujar_panel(ax, 1, nJ1, fJ1, 0.0, anot[1])

ax.set_xlim(-4.9, 3.9); ax.set_ylim(-2.6, GAP+9+2.6)

# flecha E con gradiente + cabeza
grad = LinearSegmentedColormap.from_list("e", ["#eef4fc", "#a8c6ea", "#5f93d4"])
ax.imshow(np.linspace(0,1,256).reshape(-1,1), aspect="auto", cmap=grad,
          extent=[-4.4,-4.2, -1.6, GAP+9], origin="lower", zorder=1, clip_on=False)
ax.add_patch(Polygon([[-4.5,GAP+9],[-4.1,GAP+9],[-4.3,GAP+10.4]], closed=True,
             fc="#5f93d4", ec="none", zorder=1, clip_on=False))
ax.text(-4.3, GAP+11.1, "E", fontsize=14, ha="center", va="center",
        fontstyle="italic", fontweight="bold")

# cajas J y THz
def jbox(y,txt):
    ax.text(-3.55, y, txt, fontsize=12, ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="square,pad=0.32", fc="white", ec="black"))
jbox(GAP+4.5, "J = 2"); jbox(2.5, "J = 1")
ax.annotate("", xy=(-3.55, GAP+2.6), xytext=(-3.55, 4.4),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5), zorder=4)
ax.text(-3.72, (GAP+2.6+4.4)/2, f"{anot['dJ_THz']:.2f} THz", fontsize=11,
        ha="right", va="center", rotation=90)

# xi header + titulo
ax.text(3.35, GAP+9.3, "\u03be", fontsize=15, ha="center", fontstyle="italic")
fig.text(0.17, 0.955, "(a)", fontsize=14, fontweight="bold", ha="left")
fig.text(0.225, 0.955, "energy (not to scale)", fontsize=12, ha="left")

plt.savefig("fig2a.png", dpi=150, bbox_inches="tight")
print("ok")