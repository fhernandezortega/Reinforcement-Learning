import json, numpy as np
import os
import numpy as np, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.rlqls_env_cah import RLQLSEnvCaH
from rl.dqn import DQNAgent

INSET_CKPT = 1000          # el modelo convergido (media 8.4)

env = RLQLSEnvCaH(max_steps=200); env.reset(seed=10002)  # eval_env de seed 2
ag = DQNAgent(env.n_states, env.n_actions, N_training=1000, use_qmdp=True,
              purity_threshold=env.purity_threshold)
ag.load(f"checkpoints/model_ep{INSET_CKPT}.pt")

L = []
for _ in range(1000):
    s, _ = env.reset(); d = False; st = 0
    while not d:
        a = ag.select_action(s, explore=False)
        s, r, term, trunc, _ = env.step(a); st += 1; d = term or trunc
    L.append(st)
L = np.array(L)

# reescribir inset en el JSON existente
d = json.load(open("fig2c_data.json"))
d["inset_lengths"] = L.tolist()
d["inset_at"] = INSET_CKPT
json.dump(d, open("fig2c_data.json", "w"))
print(f"inset regenerado desde model_ep{INSET_CKPT}: media={L.mean():.1f}, "
      f">100={int((L>100).sum())}, <=15={int((L<=15).sum())}")