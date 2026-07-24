"""
fig_s5.py — Fig. S5: distribucion de estados terminales (arriba) y de pulsos
elegidos (abajo) en los episodios de testing del "model 600".
Reproduce el ~62% terminando en |J,-J+1/2,-> o |J,-J-1/2,->.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl.decision_tree import load_policy       # importa torch dentro; sin ete3 (sin conflicto)
try:
    from env.rlqls_env_cah import RLQLSEnvCaH
except ModuleNotFoundError:
    from env.rlqls_env_cah import RLQLSEnvCaH

ROMAN = ['I','II','III','IV','V','VI','VII','VIII','IX','X',
         'XI','XII','XIII','XIV','XV','XVI']


def run_episodes(env, policy, n=2000, seed=321):
    env.reset(seed=seed)
    term_states = []          # indice del estado terminal (solo episodios exitosos)
    pulse_counts = np.zeros(env.n_actions, dtype=np.int64)
    for _ in range(n):
        s, _ = env.reset(); done = False
        while not done:
            a = policy(s.astype(np.float32))
            pulse_counts[a] += 1
            s, r, term, trunc, info = env.step(a); done = term or trunc
        if term:                                   # termino por pureza
            term_states.append(int(np.asarray(s).argmax()))
    return np.array(term_states), pulse_counts


if __name__ == "__main__":
    env = RLQLSEnvCaH()
    ck = "checkpoints/model_ep600.pt"
    policy = load_policy(ck, env.n_states, env.n_actions)
    print(f"Modelo: {ck}")

    term_states, pulse_counts = run_episodes(env, policy, n=2000)
    N = len(term_states)

    # distribucion de estados terminales (normalizada)
    state_dist = np.bincount(term_states, minlength=env.n_states) / N
    # distribucion de pulsos (normalizada)
    pulse_dist = pulse_counts / pulse_counts.sum()

    # --- check del 62%: |J,-J+1/2,-> y |J,-J-1/2,-> = IV,V (J=1) y XII,XIII (J=2) ---
    targets = []
    for i, (J, m, xi) in enumerate(env.ham.eig_labels):
        if xi == '-' and (abs(m - (-J + 0.5)) < 1e-6 or abs(m - (-J - 0.5)) < 1e-6):
            targets.append(i)
    frac_target = state_dist[targets].sum()
    print(f"Episodios exitosos: {N}")
    print(f"Estados objetivo |J,-J±1/2,->: {[ROMAN[i] for i in targets]}")
    print(f"Fraccion terminando ahi: {100*frac_target:.1f}%  (paper: 62%)")

    # etiquetas del eje x (arriba)
    xlabels = [f"{ROMAN[i]}: |{J},{m:+.1f},{xi}>"
               for i, (J, m, xi) in enumerate(env.ham.eig_labels)]

    # ---- figura ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))

    ax1.bar(range(env.n_states), state_dist, color="green", edgecolor="k", linewidth=0.4)
    ax1.set_xticks(range(env.n_states))
    ax1.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=6.5)
    ax1.set_ylabel("state distribution")
    ax1.set_title(f"terminal pure state  (model 600)   —   "
                  f"{100*frac_target:.0f}% en |J,-J±1/2,->")

    ax2.bar(range(1, env.n_actions + 1), pulse_dist,
            color="olive", edgecolor="k", linewidth=0.4)
    ax2.set_xticks(range(1, env.n_actions + 1))
    ax2.set_xlabel("pulse choices")
    ax2.set_ylabel("action distribution")

    plt.tight_layout()
    out = "fig_s5.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Figura guardada: {out}")