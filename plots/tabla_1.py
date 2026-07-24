"""
table_1.py — Tabla I: % de episodios exitosos vs numero de pulsos aplicados.
Fila RL: fraccion acumulada P(longitud <= N) de los episodios de test del
modelo final (episode_lengths.npy). Fila sweeping: requiere el baseline (aparte).
"""
import os, numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def rl_row(lengths, pulses=range(2, 19)):
    lengths = np.asarray(lengths)
    return {n: 100.0 * np.mean(lengths <= n) for n in pulses}

def load_or_make_lengths(n_eval=5000, ckpt="checkpoints/model_ep1000.pt"):
    path = "episode_lengths.npy"
    if os.path.exists(path):
        return np.load(path)
    # generar de un checkpoint
    import torch  # (sin ete3 aqui, no hay conflicto)
    from rl.decision_tree import load_policy
    try:
        from env.rlqls_env_cah import RLQLSEnvCaH
    except ModuleNotFoundError:
        from env.rlqls_env_cah import RLQLSEnvCaH
    env = RLQLSEnvCaH(); env.reset(seed=777)
    policy = load_policy(ckpt, env.n_states, env.n_actions)
    L = []
    for _ in range(n_eval):
        s, _ = env.reset(); done = False; steps = 0
        while not done:
            a = policy(s.astype(np.float32))
            s, r, term, trunc, info = env.step(a); done = term or trunc; steps += 1
        L.append(steps)
    L = np.asarray(L, dtype=np.int32)
    np.save(path, L)
    return L


if __name__ == "__main__":
    L = load_or_make_lengths()
    row = rl_row(L)
    # referencia del paper (fila RL)
    paper = {2:0,3:15,4:35,5:35,6:35,7:45,8:56,18:99}

    print(f"Modelo final: {len(L)} episodios de test | media={L.mean():.2f} pasos "
          f"| exito@30={100*np.mean(L<30):.1f}%\n")
    print("TABLA I — fila RL: % de episodios terminados en <= N pulsos")
    header = "  ".join(f"{n:>3}" for n in row)
    print(f"  # pulsos : {header}")
    print(f"  RL (FH&LF): " + "  ".join(f"{row[n]:>3.0f}" for n in row))
    print(f"  RL(paper): " + "  ".join(
        f"{paper.get(n,''):>3}" if n in paper else "  ." for n in row))
    print(f"\n  (paper reporta 2→0, 3→15, 4→35, ..., 8→56, ..., 18→99 %)")