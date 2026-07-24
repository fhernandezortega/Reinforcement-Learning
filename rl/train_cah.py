"""
train_cah.py — Entrenamiento RL-QLS CaH+ (Fig. 2b y 2c) — versión unificada.

Funcion principal:
  train_all()  -> UNA corrida genera TODO:
       - training_history.json  (Fig. 2b; de la semilla primaria=seeds[0])
       - checkpoints/model_ep<N>.pt  (para fig_2c.py basado en checkpoints)
       - fig2c_data.json  (Fig. 2c multi-semilla: curva promedio + banda + inset)

Funciones auxiliares (mismas de antes, por si quieres una sola cosa):
  train()       -> solo un modelo (history + checkpoints)
  train_fig2c() -> solo la curva multi-semilla (fig2c_data.json)

Salida: directorio OUTDIR (env RLQLS_OUT, por defecto el directorio actual).

max_steps: tope de pasos por episodio. El paper no especifica ninguno; un tope
bajo trunca los episodios largos del inicio del entrenamiento y sesga la media
hacia abajo (con tope 30 se trunca el 3.2%; con 200, el 0%). Default 200.
"""
import os, sys, json, time
import numpy as np, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from env.rlqls_env_cah import RLQLSEnvCaH
except ModuleNotFoundError:
    from env.rlqls_env_cah import RLQLSEnvCaH
try:
    from rl.dqn import DQNAgent
except ModuleNotFoundError:
    from rl.dqn import DQNAgent

OUTDIR = os.environ.get("RLQLS_OUT", ".")


def _make_agent(env, n_episodes, use_qmdp, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    return DQNAgent(
        n_states=env.n_states, n_actions=env.n_actions,
        N_training=n_episodes, use_qmdp=use_qmdp,
        purity_threshold=env.purity_threshold,
        loss_type='smooth_l1', eps_end=0.005, lr=5e-4, tau_update=0.001,
        batch_size=32, device='cpu',
    )


def evaluate(env, agent, n=200, return_raw=False):
    """Longitud de episodio con politica greedy (explore=False) — testing Fig.2c."""
    L = []
    for _ in range(n):
        s, _ = env.reset(); done = False; steps = 0
        while not done:
            a = agent.select_action(s, explore=False)
            s, r, term, trunc, info = env.step(a)
            done = term or trunc; steps += 1
        L.append(steps)
    m = float(np.mean(L))
    return (m, L) if return_raw else m

def _run_seed(seed, n_episodes, use_qmdp, eval_every, eval_n,
              save_ckpt=False, ckpt_dir=None, inset_at=None, inset_n=0,
              final_eval_n=0, max_steps=200,tag=""):
    """Entrena una semilla; devuelve (lengths, eval_at, eval_len, inset_lengths, final_lengths).
    final_lengths: distribucion de longitudes del modelo FINAL sobre final_eval_n
    episodios de test (para la Tabla I). Vacia si final_eval_n=0."""
    env = RLQLSEnvCaH(max_steps=max_steps); env.reset(seed=seed)
    eval_env = RLQLSEnvCaH(max_steps=max_steps); eval_env.reset(seed=10_000 + seed)
    agent = _make_agent(env, n_episodes, use_qmdp, seed)
    lengths = []; eval_at = []; eval_len = []; inset_lengths = None
    for ep in range(n_episodes):
        s, _ = env.reset(); done = False; steps = 0
        while not done:
            a = agent.select_action(s, explore=True)
            if use_qmdp:
                p0, p1, S0, S1, _, _ = env.qmdp_branches(s, a)
                s2, r, term, trunc, info = env.step(a)
                agent.store(s, a, r, s2, term, p0, p1, S0, S1)
            else:
                s2, r, term, trunc, info = env.step(a)
                agent.store(s, a, r, s2, term)
            agent.update(); s = s2; steps += 1; done = term or trunc
        agent.decay_epsilon(ep + 1); lengths.append(steps)
        if (ep + 1) % eval_every == 0:
            eval_at.append(ep + 1)
            eval_len.append(evaluate(eval_env, agent, n=eval_n))
            if save_ckpt:
                agent.save(os.path.join(ckpt_dir, f"model_ep{ep+1}.pt"))
        if inset_at is not None and (ep + 1) == inset_at:
            _, inset_lengths = evaluate(eval_env, agent, n=inset_n, return_raw=True)
    # distribucion del modelo final (Tabla I)
    final_lengths = None
    if final_eval_n > 0:
        _, final_lengths = evaluate(eval_env, agent, n=final_eval_n, return_raw=True)
    return lengths, eval_at, eval_len, inset_lengths, final_lengths


# ---------------------------------------------------------------------------
# UNIFICADA: genera history + checkpoints + fig2c_data en una corrida
# ---------------------------------------------------------------------------
def train_all(seeds=(0, 1, 2, 3, 4), n_episodes=1000, use_qmdp=True,
              eval_every=50, eval_n=200, inset_at=600, inset_n=1000,
              final_eval_n=5000, max_steps=200):
    os.makedirs(OUTDIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTDIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    t0 = time.time()
    print(f"=== train_all | {len(seeds)} semillas x {n_episodes} ep "
          f"| {'qMDP' if use_qmdp else 'MDP'} ===")

    curves = []
    hist = {}
    inset_lengths = None
    final_lengths = None
    for si, seed in enumerate(seeds):
        primary = (si == 0)
        lengths, eval_at, eval_len, inset, finl = _run_seed(
            seed, n_episodes, use_qmdp, eval_every, eval_n,
            save_ckpt=primary, ckpt_dir=ckpt_dir,
            inset_at=(inset_at if primary else None), inset_n=inset_n,
            final_eval_n=(final_eval_n if primary else 0), max_steps=max_steps)
        curves.append(eval_len)
        if primary:
            hist = {"steps": lengths, "eval_at": eval_at, "eval_len": eval_len}
            inset_lengths = inset
            final_lengths = finl
        print(f"  semilla {seed}{' (primaria)' if primary else ''}: "
              f"eval_final={eval_len[-1]:.2f} | {time.time()-t0:.0f}s")

    curves = np.array(curves)                      # (K, n_checkpoints)

    # 1) training_history.json  (Fig. 2b)
    with open(os.path.join(OUTDIR, "training_history.json"), "w") as f:
        json.dump({**hist, "sweeping": 9.7, "max_steps": max_steps,
                   "mode": "qMDP" if use_qmdp else "MDP", "seed": seeds[0]}, f)
    # 2) fig2c_data.json  (Fig. 2c multi-semilla)
    with open(os.path.join(OUTDIR, "fig2c_data.json"), "w") as f:
        json.dump({"eval_at": hist["eval_at"], "curves": curves.tolist(),
                   "mean": curves.mean(0).tolist(), "std": curves.std(0).tolist(),
                   "inset_at": inset_at, "inset_lengths": inset_lengths,
                   "sweeping": 9.7, "n_seeds": len(seeds), "eval_n": eval_n}, f)
    # 3) episode_lengths.npy  (modelo final, semilla primaria -> Tabla I)
    if final_lengths is not None:
        np.save(os.path.join(OUTDIR, "episode_lengths.npy"),
                np.asarray(final_lengths, dtype=np.int32))

    print(f"\nGenerado en '{OUTDIR}':")
    print(f"  training_history.json   -> Fig. 2b (semilla {seeds[0]})")
    print(f"  checkpoints/            -> {len(hist['eval_at'])} modelos (semilla {seeds[0]}) para tu fig_2c.py")
    print(f"  fig2c_data.json         -> Fig. 2c multi-semilla (mean final = {curves.mean(0)[-1]:.2f}, paper ~8.3)")
    if final_lengths is not None:
        arr = np.asarray(final_lengths)
        print(f"  episode_lengths.npy     -> Tabla I ({len(arr)} test eps, "
              f"exito={100*np.mean(arr < 30):.1f}%, media={arr.mean():.2f})")
    print(f"Tiempo total: {(time.time()-t0)/60:.1f} min")

# ---- envoltorios de una sola cosa (compatibilidad) ----
def train(n_episodes=1000, use_qmdp=True, seed=0, eval_every=50,
          final_eval_n=5000, max_steps=200):
    os.makedirs(OUTDIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTDIR, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
    lengths, eval_at, eval_len, _, final_lengths = _run_seed(
        seed, n_episodes, use_qmdp, eval_every, 200,
        save_ckpt=True, ckpt_dir=ckpt_dir, final_eval_n=final_eval_n,
        max_steps=max_steps)
    with open(os.path.join(OUTDIR, "training_history.json"), "w") as f:
        json.dump({"steps": lengths, "eval_at": eval_at, "eval_len": eval_len,
                   "sweeping": 9.7, "max_steps": max_steps,
                   "mode": "qMDP" if use_qmdp else "MDP", "seed": seed}, f)
    if final_lengths is not None:
        np.save(os.path.join(OUTDIR, "episode_lengths.npy"),
                np.asarray(final_lengths, dtype=np.int32))
    print("Generado: training_history.json + checkpoints/ + episode_lengths.npy")


if __name__ == "__main__":
    # UNA corrida -> Fig. 2b y Fig. 2c (ambas versiones)
    train_all(seeds=(0, 1, 2, 3, 4), n_episodes=1000,
              eval_every=50, eval_n=200, inset_at=600, inset_n=1000,
              max_steps=200)