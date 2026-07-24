"""
decision_tree.py — Extractor del arbol de decision de la politica RL-QLS (Fig. 2d / S4).

El arbol es el despliegue determinista de la politica greedy:
  - cada nodo = estado de poblacion S
  - accion del nodo = pulso greedy  argmax_a Q(S)   (numero rojo en el paper)
  - dos ramas = resultados de medida k=0, k=1 con probabilidades p0, p1
                (Ec. 4a; numeros negros). Reusa env.qmdp_branches().
  - hoja terminal = estado puro (max P_J > umbral). En el paper (S4) las ramas
    que terminan se omiten; aqui se marcan y opcionalmente se podan del dibujo.

Poda: profundidad maxima y probabilidad minima de rama (como hace el paper
implicitamente al no mostrar ramas de probabilidad despreciable).
"""
import sys

import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_policy(ckpt_path, n_states, n_actions, hidden=(128, 128, 128)):
    import torch
    try:
        from rl.dqn import QNetwork
    except ModuleNotFoundError:
        from rl.dqn import QNetwork
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = QNetwork(n_states=n_states, n_actions=n_actions, hidden_dims=list(hidden))
    sd = ckpt["q_online"] if isinstance(ckpt, dict) and "q_online" in ckpt else ckpt
    net.load_state_dict(sd); net.eval()

    def policy(S):
        with torch.no_grad():
            return int(net(torch.as_tensor(S, dtype=torch.float32)).argmax().item())
    return policy


ROMAN = ['I','II','III','IV','V','VI','VII','VIII','IX','X',
         'XI','XII','XIII','XIV','XV','XVI']


def extract_tree(env, policy, max_depth=8, min_prob=0.02, merge=False, round_dec=3):
    """
    Devuelve (nodes, root_id).
      nodes[id] = {id, state, action(1-based o None), children[(child,prob,k)],
                   terminal, depth, purity, peak(label)}
    merge=True fusiona estados identicos (produce los nodos multi-hijo del paper),
    pero puede crear DAG/ciclos; para un arbol ETE limpio usar merge=False.
    """
    nodes = {}; sig2id = {}; counter = [0]
    labels = env.ham.eig_labels

    def peak_label(S):
        j = int(S.argmax())
        J, m, xi = labels[j]
        return f"{ROMAN[j]}:|{J},{m:+.1f},{xi}>"

    def sig(S): return tuple(np.round(S, round_dec))

    def build(S, depth, ancestors):
        if merge:
            k = sig(S)
            if k in sig2id:
                return sig2id[k]
        nid = counter[0]; counter[0] += 1
        if merge: sig2id[sig(S)] = nid
        node = {"id": nid, "state": S.copy(), "action": None, "children": [],
                "terminal": False, "depth": depth,
                "purity": float(S.max()), "peak": peak_label(S)}
        nodes[nid] = node

        if S.max() > env.purity_threshold or depth >= max_depth:
            node["terminal"] = True
            return nid

        a = policy(S.astype(np.float32))
        node["action"] = int(a) + 1     # 1-based, como el paper
        p0, p1, S0, S1, t0, t1 = env.qmdp_branches(S, a)
        for (p, Sc, kk) in [(p0, np.asarray(S0, float), 0),
                            (p1, np.asarray(S1, float), 1)]:
            if p < min_prob:
                continue
            # evita ciclos con merge: no vuelvas a un ancestro
            if merge and sig(Sc) in ancestors:
                continue
            cid = build(Sc, depth + 1, ancestors | {sig(S)})
            node["children"].append({"child": cid, "prob": float(p), "k": kk})
        return nid

    S_init = env.ham.get_boltzmann(env.T).astype(np.float64)
    root = build(S_init, 0, set())
    return nodes, root


def tree_stats(nodes, root, purity_threshold):
    leaves = [x for x in nodes.values() if x["terminal"]]
    pure   = [x for x in leaves if x["purity"] > purity_threshold]
    cut    = [x for x in leaves if x["purity"] <= purity_threshold]   # max_depth
    ok = all(abs(sum(c["prob"] for c in x["children"]) - 1.0) <= 0.05
             for x in nodes.values()
             if not x["terminal"] and len(x["children"]) == 2)
    return dict(n_nodes=len(nodes), n_leaves=len(leaves),
                n_pure_leaves=len(pure), n_cut_leaves=len(cut),
                max_depth=max((x["depth"] for x in leaves), default=0),
                branches_sum_1=ok,
                multi_offspring=sum(1 for x in nodes.values()
                                    if len(x["children"]) > 2))

def terminal_distribution(nodes, root):
    """Distribucion de estados terminales PONDERADA por probabilidad de camino
    (comparable con la Fig. S5). Contar nodos sobreestima las ramas raras."""
    dist = {}
    def rec(nid, p):
        nd = nodes[nid]
        if nd["terminal"]:
            dist[nd["peak"]] = dist.get(nd["peak"], 0.0) + p
            return
        for c in nd["children"]:
            rec(c["child"], p * c["prob"])
    rec(root, 1.0)
    return dict(sorted(dist.items(), key=lambda kv: -kv[1]))

def print_tree(nodes, root, max_lines=40):
    lines = [0]
    def rec(nid, prefix, edge):
        if lines[0] >= max_lines: return
        nd = nodes[nid]
        if nd["terminal"]:
            print(f"{prefix}{edge}[TERM {nd['peak']} P={nd['purity']:.2f}]")
        else:
            print(f"{prefix}{edge}pulso {nd['action']}  ({nd['peak']})")
        lines[0] += 1
        for i, c in enumerate(nd["children"]):
            last = (i == len(nd["children"]) - 1)
            e = f"└─k={c['k']}(p={c['prob']:.2f})─ " if last else f"├─k={c['k']}(p={c['prob']:.2f})─ "
            rec(c["child"], prefix + ("    " if last else "│   "), e)
    rec(root, "", "")


if __name__ == "__main__":
    from env.rlqls_env_cah import RLQLSEnvCaH

    env = RLQLSEnvCaH()
    ck = "checkpoints/model_ep1000.pt"
    policy = load_policy(ck, env.n_states, env.n_actions)
    print(f"Politica cargada: {ck}\n")

    # --- arbol truncado (Fig. 2d) ---
    nodes, root = extract_tree(env, policy, max_depth=8, min_prob=0.02, merge=False)
    print("Estadisticas del arbol:")
    for k, v in tree_stats(nodes, root, env.purity_threshold).items():
        print(f"  {k}: {v}")

    print("\nArbol (primeros niveles):")
    print_tree(nodes, root, max_lines=30)

    # --- arbol completo (Fig. S4) para la distribucion terminal ---
    nodes_c, root_c = extract_tree(env, policy, max_depth=25, min_prob=0.0)
    dist = terminal_distribution(nodes_c, root_c)

    print("\nDistribucion terminal ponderada (Fig. S5):")
    for lab, p in list(dist.items())[:8]:
        print(f"  {lab:22s} {100*p:5.1f}%")

    tgt = [f"{ROMAN[env.ham.state_index(J, m, '-')]}:"
           for J in (1, 2) for m in (-J + 0.5, -J - 0.5)]
    s = sum(p for lab, p in dist.items() if any(lab.startswith(t) for t in tgt))
    print(f"  -> en |J,-J±1/2,->: {100*s:.0f}%   (Fig. S5: ~60%)")