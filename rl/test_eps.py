import numpy as np, torch, random
import numpy as np, sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.rlqls_env_cah import RLQLSEnvCaH
from rl.dqn import DQNAgent

np.random.seed(2); torch.manual_seed(2); random.seed(2)
env = RLQLSEnvCaH(max_steps=200); env.reset(seed=2)
ag = DQNAgent(env.n_states, env.n_actions, N_training=400, use_qmdp=True,
              purity_threshold=env.purity_threshold, eps_end=0.005,
              lr=5e-4, tau_update=0.001, batch_size=32)

# imprime epsilon en varios puntos para VER si es negativo
print("epsilon a lo largo del entrenamiento:")
for n in [0, 50, 100, 200, 400]:
    print(f"  ep {n}: eps = {ag._compute_eps(n):.4f}")

L = []
for ep in range(400):
    s, _ = env.reset(); done = False; st = 0
    while not done:
        a = ag.select_action(s, explore=True)
        p0, p1, S0, S1, _, _ = env.qmdp_branches(s, a)
        s2, r, term, trunc, _ = env.step(a)
        ag.store(s, a, r, s2, term, p0, p1, S0, S1); ag.update(); s = s2; st += 1
        done = term or trunc
    ag.decay_epsilon(ep + 1); L.append(st)

L = np.array(L)
print(f"\nlongitud media ultimos 100 ep: {L[-100:].mean():.1f}")
print(f"  (con epsilon correcto converge a ~8; con S16 rota se queda alto)")