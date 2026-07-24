import os
import json
import matplotlib.pyplot as plt


# ==========================================
# Rutas
# ==========================================

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "tree_data_structured.json")
OUT_PATH  = os.path.join(HERE, "fig2d_tree.png")


# ==========================================
# Parametros
# ==========================================

DX = 1.8          # separacion horizontal por nivel
DY = 1.0          # separacion vertical entre hojas
MIN_BRANCH_PROB = 0.02   # podar ramas de baja probabilidad acumulada


with open(JSON_PATH, "r") as f:
    tree = json.load(f)


# ==========================================
# 1) Layout: asignar (x, y) sin solapes
#    x = profundidad, y = posicion segun hojas
# ==========================================

nodes = []   # lista de dicts: {id, x, y, node}
edges = []   # lista de dicts: {u, v, kind, prob}
_next_id = [0]
_leaf_y  = [0.0]

def kept_children(node):
    """Hijos que superan el umbral de probabilidad acumulada."""
    out = []
    for kind in ("k=0", "k=1"):
        ch = node.get("children", {}).get(kind)
        if ch is not None and ch.get("probability", 0.0) >= MIN_BRANCH_PROB:
            out.append((kind, ch))
    return out

def layout(node):
    """Asigna posiciones; devuelve (id, y) del nodo."""
    nid = _next_id[0]; _next_id[0] += 1
    x = node["depth"] * DX

    children = kept_children(node) if node["type"] == "node" else []

    if not children:
        y = _leaf_y[0]; _leaf_y[0] -= DY
    else:
        child_ys = []
        for kind, ch in children:
            cid, cy = layout(ch)
            edges.append({
                "u": nid, "v": cid, "kind": kind,
                "prob": ch.get("branch_prob", 0.0),
            })
            child_ys.append(cy)
        y = sum(child_ys) / len(child_ys)

    nodes.append({"id": nid, "x": x, "y": y, "node": node})
    return nid, y

layout(tree)

pos = {n["id"]: (n["x"], n["y"]) for n in nodes}


# ==========================================
# 2) Dibujar
# ==========================================

fig, ax = plt.subplots(figsize=(13, 8))

# aristas
for e in edges:
    x0, y0 = pos[e["u"]]
    x1, y1 = pos[e["v"]]
    color = "black" if e["kind"] == "k=0" else "deepskyblue"
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                        shrinkA=12, shrinkB=12),
        zorder=1,
    )
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(xm, ym + 0.10, f"{e['prob']:.2f}", color=color,
            fontsize=8, fontweight="bold", ha="center", zorder=3)

# nodos
for n in nodes:
    node = n["node"]
    x, y = n["x"], n["y"]
    if node["type"] == "terminal":
        label = str(node["terminal_state"])
        fc, ec = "#D9D9D9", "gray"
    elif node["type"] == "cutoff":
        label = "…"
        fc, ec = "white", "orange"
    else:
        label = str(node["action"])   # numero de pulso (rojo)
        fc, ec = "white", "red"
    ax.scatter([x], [y], s=650, facecolors=fc, edgecolors=ec,
               linewidths=1.6, zorder=2)
    ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=4)

# leyenda estilo paper
y_top = max(p[1] for p in pos.values()) + 1.0
ax.text(-0.4, y_top,       "k=0  →  termination", color="black",
        fontsize=11, va="center")
ax.text(-0.4, y_top - 0.6, "k=1", color="deepskyblue",
        fontsize=11, va="center")

ax.set_title("Fig. 2(d) — RL-QLS Decision Tree", fontsize=15)
ax.axis("off")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Figura guardada: {OUT_PATH}", flush=True)
plt.show()