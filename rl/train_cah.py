import os, sys, json, time
import numpy as np, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.rlqls_env_cah import RLQLSEnvCaH
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
              final_eval_n=0, max_steps=200, tag="", A_matrices=None):
    env = RLQLSEnvCaH(max_steps=max_steps, A_matrices=A_matrices); env.reset(seed=seed)
    eval_env = RLQLSEnvCaH(max_steps=max_steps, A_matrices=A_matrices); eval_env.reset(seed=10_000 + seed)
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
    final_lengths = None
    if final_eval_n > 0:
        _, final_lengths = evaluate(eval_env, agent, n=final_eval_n, return_raw=True)
    return lengths, eval_at, eval_len, inset_lengths, final_lengths


def train_all(seeds=(0, 1, 2, 3, 4), n_episodes=1000, use_qmdp=True,
              eval_every=10, eval_n=200, inset_at=1000, inset_n=1000,
              final_eval_n=5000, max_steps=200, primary_seed=None):
    os.makedirs(OUTDIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTDIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    if primary_seed is None:
        primary_seed = seeds[0]
    if primary_seed not in seeds:
        seeds = (primary_seed,) + tuple(seeds)
    t0 = time.time()
    print(f"=== train_all | {len(seeds)} semillas x {n_episodes} ep "
          f"| {'qMDP' if use_qmdp else 'MDP'} | max_steps={max_steps} "
          f"| primaria={primary_seed} ===")

    _probe = RLQLSEnvCaH(max_steps=max_steps)   # una sola corrida del solver
    A_matrices = (_probe.A0, _probe.A1)         # inyectadas a todos los envs/semillas

    curves = []; hist = {}; inset_lengths = None; final_lengths = None
    for si, seed in enumerate(seeds):
        primary = (seed == primary_seed)
        lengths, eval_at, eval_len, inset, finl = _run_seed(
            seed, n_episodes, use_qmdp, eval_every, eval_n,
            save_ckpt=primary, ckpt_dir=ckpt_dir,
            inset_at=(inset_at if primary else None), inset_n=inset_n,
            final_eval_n=(final_eval_n if primary else 0), max_steps=max_steps,
            A_matrices=A_matrices)
        curves.append(eval_len)
        if primary:
            hist = {"steps": lengths, "eval_at": eval_at, "eval_len": eval_len}
            inset_lengths = inset
            final_lengths = finl
        print(f"  semilla {seed}{' (primaria)' if primary else ''}: "
              f"eval_final={eval_len[-1]:.2f} | max_len={max(lengths)} "
              f"| {time.time()-t0:.0f}s")
        # guardar el history de CADA semilla -> fig2b.py / fig2b1.py
        with open(os.path.join(OUTDIR, f"training_history_seed{seed}.json"), "w") as f:
            json.dump({"steps": lengths, "sweeping": 10.0, "max_steps": max_steps,
                       "mode": "qMDP" if use_qmdp else "MDP", "seed": seed}, f)

    curves = np.array(curves)

    with open(os.path.join(OUTDIR, "training_history.json"), "w") as f:
        json.dump({**hist, "sweeping": 10.0, "max_steps": max_steps,
                   "mode": "qMDP" if use_qmdp else "MDP", "seed": primary_seed}, f)
    with open(os.path.join(OUTDIR, "fig2c_data.json"), "w") as f:
        json.dump({"eval_at": hist["eval_at"], "curves": curves.tolist(),
                   "mean": curves.mean(0).tolist(), "std": curves.std(0).tolist(),
                   "inset_at": inset_at, "inset_lengths": inset_lengths,
                   "sweeping": 10.0, "n_seeds": len(seeds), "eval_n": eval_n,
                   "max_steps": max_steps}, f)
    if final_lengths is not None:
        np.save(os.path.join(OUTDIR, "episode_lengths.npy"),
                np.asarray(final_lengths, dtype=np.int32))

    print(f"\nGenerado en '{OUTDIR}':")
    print(f"  training_history_seed*.json -> {len(seeds)} archivos (fig2b.py / fig2b1.py)")
    print(f"  training_history.json       -> Fig. 2b (semilla {primary_seed})")
    print(f"  checkpoints/                -> {len(hist['eval_at'])} modelos (semilla {primary_seed})")
    print(f"  fig2c_data.json             -> Fig. 2c (mean final = {curves.mean(0)[-1]:.2f}, paper ~8.3)")
    if final_lengths is not None:
        arr = np.asarray(final_lengths)
        print(f"  episode_lengths.npy         -> Tabla I ({len(arr)} test eps, "
              f"exito={100*np.mean(arr < max_steps):.1f}%, media={arr.mean():.2f})")
    print(f"Tiempo total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    train_all(seeds=(0, 1, 2, 3, 4), n_episodes=1000,
              eval_every=10, eval_n=200, inset_at=1000, inset_n=1000,
              max_steps=200, primary_seed=2)