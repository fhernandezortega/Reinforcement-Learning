"""
qmdp.py — Update de valor qMDP (RL-QLS, Ec. S18) para el agente DQN.

Diferencia MDP (Ec. S17) vs qMDP (Ec. S18):
  MDP :  y = R + gamma * max_a Q(S_{t+1}, a)          con S_{t+1} MUESTREADO
  qMDP:  y = R + gamma * [ p0 * max_a Q(S'_{k=0}, a)
                         + p1 * max_a Q(S'_{k=1}, a) ] con AMBAS ramas deterministas

El qMDP promedia explicitamente sobre los dos resultados de medida ponderados
por p0,p1 (POVM, Ecs. S2-S3). No muestrea k para el target -> menor varianza y,
para Hilbert grandes (H3O+), imprescindible (loss ~3 ordenes menor, Fig. S8).

gamma = 1 (paper). R = -1 por paso. Cada rama que ya es pura NO hace bootstrap.

Se provee:
  - qmdp_targets_np : referencia NumPy (para validar la matematica).
  - qmdp_loss_torch : version PyTorch lista para el bucle de entrenamiento (doble-Q).
"""
import sys

import numpy as np
import os
# Añade el directorio padre (raíz del proyecto) al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.rlqls_env_cah import RLQLSEnvCaH

# ---------------------------------------------------------------------------
# Ramas deterministas vectorizadas (Ec. 4a-4b para las dos k, sin muestrear)
# ---------------------------------------------------------------------------
def qmdp_branch_batch(A0, A1, states, actions, purity_threshold):
    """
    A0,A1: (nA,NS,NS) ; states:(B,NS) ; actions:(B,)
    Devuelve p0,p1 (B,) ; S0,S1 (B,NS) normalizados ; term0,term1 (B,) bool.
    """
    A0a = A0[actions]; A1a = A1[actions]            # (B,NS,NS)
    s0 = np.einsum('bij,bj->bi', A0a, states)       # (B,NS)
    s1 = np.einsum('bij,bj->bi', A1a, states)
    p0 = s0.sum(1); p1 = s1.sum(1)
    tot = p0 + p1
    p0 = np.where(tot > 1e-12, p0/np.where(tot>0,tot,1), 0.5)
    p1 = 1.0 - p0
    S0 = s0 / (s0.sum(1, keepdims=True) + 1e-300)
    S1 = s1 / (s1.sum(1, keepdims=True) + 1e-300)
    term0 = S0.max(1) > purity_threshold
    term1 = S1.max(1) > purity_threshold
    return p0, p1, S0, S1, term0, term1


# ---------------------------------------------------------------------------
# Target qMDP (referencia NumPy). q_fn(states)->(B,nA) Q-values.
# ---------------------------------------------------------------------------
def qmdp_targets_np(A0, A1, states, actions, rewards,
                    q_online_fn, q_target_fn,
                    purity_threshold, gamma=1.0, double_q=True):
    p0,p1,S0,S1,t0,t1 = qmdp_branch_batch(A0,A1,states,actions,purity_threshold)

    def bootstrap(S):
        if double_q:
            a_star = q_online_fn(S).argmax(1)              # argmax red online
            return q_target_fn(S)[np.arange(len(S)), a_star]  # valor red target
        return q_target_fn(S).max(1)

    b0 = np.where(t0, 0.0, bootstrap(S0))
    b1 = np.where(t1, 0.0, bootstrap(S1))
    y = rewards + gamma * (p0*b0 + p1*b1)                  # Ec. S18
    return y


# ---------------------------------------------------------------------------
# PyTorch: pérdida qMDP lista para el training loop (doble-Q, SmoothL1)
# ---------------------------------------------------------------------------
def qmdp_loss_torch(q_online, q_target, batch, A0_t, A1_t,
                    purity_threshold, gamma=1.0):
    """
    q_online, q_target : nn.Module  (entrada (B,NS) -> salida (B,nA))
    batch : dict con 'states'(B,NS),'actions'(B,),'rewards'(B,)  [tensores]
    A0_t, A1_t : tensores (nA,NS,NS)  (mismas matrices del entorno)
    Devuelve la pérdida SmoothL1 (Huber) del update qMDP (Ec. S18).
    """
    import torch
    S = batch['states']; a = batch['actions'].long(); R = batch['rewards']
    B = S.shape[0]

    A0a = A0_t[a]; A1a = A1_t[a]                                   # (B,NS,NS)
    s0 = torch.bmm(A0a, S.unsqueeze(-1)).squeeze(-1)              # (B,NS)
    s1 = torch.bmm(A1a, S.unsqueeze(-1)).squeeze(-1)
    p0 = s0.sum(1); p1 = s1.sum(1); tot = p0 + p1
    p0 = torch.where(tot > 1e-12, p0/tot.clamp_min(1e-12), torch.full_like(p0,0.5))
    p1 = 1.0 - p0
    S0 = s0 / (s0.sum(1, keepdim=True) + 1e-30)
    S1 = s1 / (s1.sum(1, keepdim=True) + 1e-30)
    term0 = (S0.max(1).values > purity_threshold)
    term1 = (S1.max(1).values > purity_threshold)

    with torch.no_grad():
        def boot(Sx):
            a_star = q_online(Sx).argmax(1)                       # doble-Q
            return q_target(Sx).gather(1, a_star.unsqueeze(1)).squeeze(1)
        b0 = torch.where(term0, torch.zeros(B), boot(S0))
        b1 = torch.where(term1, torch.zeros(B), boot(S1))
        y = R + gamma * (p0*b0 + p1*b1)                          # Ec. S18

    q_sa = q_online(S).gather(1, a.unsqueeze(1)).squeeze(1)
    return torch.nn.functional.smooth_l1_loss(q_sa, y)


# ======================= VALIDACION (NumPy, Q lineal ficticio) =======================
if __name__ == "__main__":
    env = RLQLSEnvCaH()
    A0 = env.A0; A1 = env.A1; NS = env.n_states; nA = env.n_actions
    rng = np.random.default_rng(0)

    # Q ficticio lineal: Q(S) = S @ W  ->  (B,nA)
    W = rng.normal(size=(NS, nA))*0.1
    q_on = lambda S: S @ W
    q_tg = lambda S: S @ (W*0.98)

    # batch de estados: Boltzmann + algunos colapsados
    s0,_ = env.reset(seed=0)
    states = [s0.astype(np.float64)]
    S = s0.astype(np.float64)
    for a in [9,4,2]:
        _,_,S0b,S1b = env._branches(S,a); S = S1b; states.append(S.copy())
    states = np.array(states)
    actions = np.array([9,4,2,10])
    rewards = np.full(len(states), -1.0)

    y = qmdp_targets_np(A0,A1,states,actions,rewards,q_on,q_tg,
                        env.purity_threshold, gamma=1.0, double_q=True)
    print("targets qMDP:", np.round(y,4))

    # comparacion MDP (muestreado) vs qMDP (esperado): media MDP ~ qMDP
    p0,p1,S0,S1,t0,t1 = qmdp_branch_batch(A0,A1,states,actions,env.purity_threshold)
    print("p0:", np.round(p0,3), "| term0:", t0, "| term1:", t1)
    # verificacion: si una rama es terminal, su bootstrap es 0
    print("\nCheck: rama terminal -> bootstrap 0, y target = R + gamma*p_otra*Q_otra")
    for i in range(len(states)):
        a_star0 = q_on(S0[i:i+1]).argmax(); a_star1=q_on(S1[i:i+1]).argmax()
        b0 = 0.0 if t0[i] else q_tg(S0[i:i+1])[0,a_star0]
        b1 = 0.0 if t1[i] else q_tg(S1[i:i+1])[0,a_star1]
        y_manual = rewards[i] + 1.0*(p0[i]*b0 + p1[i]*b1)
        print(f"  i={i}: y_vec={y[i]:.4f}  y_manual={y_manual:.4f}  match={np.isclose(y[i],y_manual)}")