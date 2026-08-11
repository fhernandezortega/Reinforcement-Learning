import os

import numpy as np, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.rlqls_env_cah import RLQLSEnvCaH
from rl.dqn import DQNAgent

env = RLQLSEnvCaH(max_steps=200); env.reset(seed=10002)
ag = DQNAgent(env.n_states, env.n_actions, N_training=1000, use_qmdp=True,
              purity_threshold=env.purity_threshold)
ag.load("checkpoints/model_ep600.pt")

L = []
for _ in range(1000):
    s, _ = env.reset(); d = False; st = 0
    while not d:
        a = ag.select_action(s, explore=False)
        s, r, term, trunc, _ = env.step(a); st += 1; d = term or trunc
    L.append(st)
L = np.array(L)
print(f"checkpoint 600: media={L.mean():.1f}, >100={int((L>100).sum())}, <=15={int((L<=15).sum())}")

ag.load("checkpoints/model_ep1000.pt")
L2 = []
for _ in range(1000):
    s, _ = env.reset(); d = False; st = 0
    while not d:
        a = ag.select_action(s, explore=False)
        s, r, term, trunc, _ = env.step(a); st += 1; d = term or trunc
    L2.append(st)
L2 = np.array(L2)
print(f"checkpoint 1000: media={L2.mean():.1f}, >100={int((L2>100).sum())}, <=15={int((L2<=15).sum())}")