"""
render_tree_ete.py — Dibuja el arbol de decision RL-QLS con ETE (Fig. 2d / S4).
Pulsos en ROJO (numeros rojos del paper), probabilidades de rama en NEGRO.
Las ramas que terminan en un estado puro se omiten (como la S4) o se muestran
como cajas azules (como la 2d), segun show_terminal.

Render a PNG requiere PyQt5. En entornos sin pantalla:  QT_QPA_PLATFORM=offscreen
"""
# IMPORTANTE (Windows): cargar torch ANTES que ete3/PyQt5. Las DLLs de Qt rompen
# la carga de torch (WinError 1114 en c10.dll) si Qt se importa primero.
import torch  # noqa: F401  (debe ir primero)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ete3 import Tree, TreeStyle, TextFace, NodeStyle, add_face_to_node

from rl.decision_tree import load_policy, extract_tree
try:
    from env.rlqls_env_cah import RLQLSEnvCaH
except ModuleNotFoundError:
    from env.rlqls_env_cah import RLQLSEnvCaH


def build_ete(nodes, root_id, show_terminal=True):
    """Convierte la estructura del extractor en un Tree de ETE."""
    def rec(nid):
        nd = nodes[nid]
        t = Tree(name=str(nid))
        t.add_feature("pulse", nd["action"])
        t.add_feature("is_term", nd["terminal"])
        t.add_feature("peak", nd["peak"])
        t.add_feature("purity", nd["purity"])
        t.add_feature("prob", None)
        t.add_feature("kmeas", None)
        for c in nd["children"]:
            child_nd = nodes[c["child"]]
            if child_nd["terminal"] and not show_terminal:
                continue                       # S4: omitir ramas terminales
            ch = rec(c["child"])
            ch.prob  = c["prob"]
            ch.kmeas = c["k"]
            t.add_child(ch)
        return t
    return rec(root_id)


def _layout(node):
    # pulso (rojo) en nodos internos
    if getattr(node, "pulse", None) is not None and not node.is_term:
        f = TextFace(str(node.pulse), fgcolor="#d00000", fsize=11, bold=True)
        add_face_to_node(f, node, column=0, position="branch-right")
    # caja azul terminal (estilo 2d)
    if getattr(node, "is_term", False):
        roman = node.peak.split(":")[0]
        bf = TextFace(f" {roman} ", fgcolor="white", fsize=9)
        bf.background.color = "#2a52be"
        bf.margin_left = 3
        add_face_to_node(bf, node, column=0, position="branch-right")
    # probabilidad de rama (negro) sobre la rama
    if getattr(node, "prob", None) is not None:
        pf = TextFace(f"{node.prob:.2f}", fgcolor="black", fsize=8)
        add_face_to_node(pf, node, column=0, position="branch-top")
    # color de la rama segun k (k=0 gris, k=1 azul) — leyenda del paper
    ns = NodeStyle()
    ns["size"] = 0
    ns["vt_line_width"] = 1
    ns["hz_line_width"] = 1
    if getattr(node, "kmeas", None) == 0:
        ns["hz_line_color"] = "#888888"
    elif getattr(node, "kmeas", None) == 1:
        ns["hz_line_color"] = "#2a52be"
    node.set_style(ns)


def render(nodes, root_id, out_png, show_terminal=True):
    tree = build_ete(nodes, root_id, show_terminal=show_terminal)
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.layout_fn = _layout
    ts.mode = "r"                    # rectangular, como el paper
    ts.branch_vertical_margin = 6
    ts.rotation = 0
    tree.render(out_png, tree_style=ts, dpi=300, w=1600)
    return tree


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model_ep1000.pt")
    ap.add_argument("--full", action="store_true",
                    help="version completa (S4): mas profundidad, sin ramas terminales")
    args = ap.parse_args()

    env = RLQLSEnvCaH()
    policy = load_policy(args.ckpt, env.n_states, env.n_actions)

    if args.full:   # Fig. S4
        nodes, root = extract_tree(env, policy, max_depth=14, min_prob=0.01, merge=False)
        out = "fig_s4_tree.png"; show_term = False
    else:           # Fig. 2d
        nodes, root = extract_tree(env, policy, max_depth=6, min_prob=0.05, merge=False)
        out = "fig_2d_tree.png"; show_term = True

    render(nodes, root, out, show_terminal=show_term)
    print(f"Arbol guardado: {out}  ({len(nodes)} nodos)")