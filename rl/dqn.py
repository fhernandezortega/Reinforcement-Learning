"""
DQN Agent — RL-QLS (PIPI2026 Sec. SD)
=======================================
Deep Q-Network con double-Q networks y experience replay para
preparacion de estados moleculares puros.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, List
import random


# =====================================================================
# QNetwork — red neuronal Q(s, a)
# =====================================================================

class QNetwork(nn.Module):
    """
    Red neuronal fully-connected para aproximar Q(s, a).

    Arquitectura del paper (Sec. SD):
      - 3 capas ocultas de 128 nodos con ReLU
      - Entrada: vector de poblacion S_t in [0,1]^NS
      - Salida: Q(s, a) para cada accion a in {0,...,NA-1}
    """

    def __init__(
        self,
        n_states:    int,
        n_actions:   int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 128]

        layers = []
        in_dim = n_states
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# ReplayBuffer — experience replay
# =====================================================================

class ReplayBuffer:
    """
    Buffer de experiencias para off-policy learning.

    Almacena (s, a, r, s', done) y, para qMDP, tambien
    (p0, p1, s'_k0, s'_k1) que permiten el update de Ec. S18.
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        p0:         float = None,
        p1:         float = None,
        next_s_k0:  np.ndarray = None,
        next_s_k1:  np.ndarray = None,
    ) -> None:
        self.buffer.append((
            state, action, reward, next_state, done,
            p0, p1, next_s_k0, next_s_k1,
        ))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        (states, actions, rewards, next_states, dones,
         p0s, p1s, next_k0s, next_k1s) = zip(*batch)

        to_t = lambda x: torch.tensor(
            np.array(x), dtype=torch.float32, device=device
        )

        states_t      = to_t(states)
        actions_t     = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_t     = to_t(rewards)
        next_states_t = to_t(next_states)
        dones_t       = to_t(dones)

        qmdp_data = None
        if p0s[0] is not None:
            qmdp_data = {
                "p0":       to_t(p0s),
                "p1":       to_t(p1s),
                "next_k0":  to_t(next_k0s),
                "next_k1":  to_t(next_k1s),
            }

        return states_t, actions_t, rewards_t, next_states_t, dones_t, qmdp_data

    def __len__(self) -> int:
        return len(self.buffer)


# =====================================================================
# DQNAgent
# =====================================================================

class DQNAgent:
    """
    Agente DQN off-policy con double-Q networks y experience replay.

    Implementa los algoritmos de PIPI2026 Sec. SD:
      1. Exploracion epsilon-greedy con decaimiento exponencial (Ec. S16)
      2. Double-Q: accion elegida por red online, valor de red target
      3. Soft target update: theta_t = tau*theta_o + (1-tau)*theta_t
      4. qMDP temporal difference (Ec. S18) cuando use_qmdp=True

    Hiperparametros del paper (Tabla S1 / Sec. SD):
      lr = 0.0005, tau_update = 0.001, eps_end = 0.005, batch_size = 32
    """

    def __init__(
        self,
        n_states:       int,
        n_actions:      int,
        hidden_dims:    Optional[List[int]] = None,
        # Replay buffer
        buffer_capacity: int   = 50_000,
        batch_size:      int   = 32,         # Sec. SD: batch size 32
        min_buffer:      int   = 512,
        # Optimizacion
        lr:              float = 5e-4,       # Tabla S1: rl=0.0005
        gamma:           float = 1.0,        # sin descuento (Sec. SD)
        loss_type:       str   = 'mse',      # 'mse' o 'smooth_l1'
        # Exploracion — Ec. S16
        eps_start:       float = 1.0,
        eps_end:         float = 0.005,      # mejor resultado Sec. SD
        N_training:      int   = 1000,       # episodios totales
        # Target network
        tau_update:      float = 0.001,      # Tabla S1: tau=0.001 (soft)
        # qMDP
        use_qmdp:        bool  = False,
        purity_threshold: float = 0.99,     # para terminal por rama (Ec. S18)
        # Dispositivo
        device:          Optional[str] = None,
    ):
        self.n_states   = n_states
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.tau_update = tau_update
        self.use_qmdp   = use_qmdp
        self.purity_threshold = purity_threshold

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Redes Q (double-Q, Sec. SD) ──────────────────────────────
        self.q_online = QNetwork(n_states, n_actions, hidden_dims).to(self.device)
        self.q_target = QNetwork(n_states, n_actions, hidden_dims).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        # ── Optimizador ───────────────────────────────────────────────
        self.optimizer = optim.Adam(self.q_online.parameters(), lr=lr)

        # ── Funcion de perdida (Sec. SD) ──────────────────────────────
        if loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        # ── Replay buffer ─────────────────────────────────────────────
        self.buffer = ReplayBuffer(buffer_capacity)

        # ── Exploracion — Ec. S16 ─────────────────────────────────────
        self.eps_start   = eps_start
        self.eps_end     = eps_end
        self.tau_eps     = 0.3 * N_training
        self._n_episodes = 0
        self.eps         = eps_start

        self._update_steps = 0

    # ------------------------------------------------------------------
    # Exploracion — Ec. S16
    # ------------------------------------------------------------------

    def _compute_eps(self, n: int) -> float:
        """eps(n) = eps_end + (eps_start - eps_end) * exp(-n / tau_eps)"""
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-n / max(self.tau_eps, 1e-8)) 
        #return self.eps_end + (self.eps_end - self.eps_start) * \
        #       np.exp(-n / max(self.tau_eps, 1e-8))

    def decay_epsilon(self, episode: Optional[int] = None) -> float:
        """
        Actualiza epsilon segun Ec. S16. LLAMAR UNA SOLA VEZ POR EPISODIO.

        - Sin argumento: usa un contador interno que se incrementa en
          cada llamada.
        - Con `episode`: fija el contador a ese numero (idempotente).
          Recomendado si tu loop ya lleva su propio indice de episodio,
          porque evita cualquier riesgo de doble decaimiento.

        `end_episode` es un ALIAS de este metodo (misma implementacion).
        No llames a ambos en el mismo episodio.
        """
        if episode is None:
            self._n_episodes += 1
        else:
            self._n_episodes = int(episode)
        self.eps = self._compute_eps(self._n_episodes)
        return self.eps

    # Alias de compatibilidad (misma implementacion, NO llamar junto con
    # decay_epsilon en el mismo episodio).
    end_episode = decay_epsilon

    # ------------------------------------------------------------------
    # Seleccion de accion
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """
        Seleccion epsilon-greedy.
        explore=True  -> entrenamiento (usa self.eps)
        explore=False -> evaluacion (greedy puro)
        """
        if explore and np.random.random() < self.eps:
            return int(np.random.randint(self.n_actions))

        with torch.no_grad():
            x = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return int(self.q_online(x).argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Memoria
    # ------------------------------------------------------------------

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        p0:         float        = None,
        p1:         float        = None,
        next_s_k0:  np.ndarray   = None,
        next_s_k1:  np.ndarray   = None,
    ) -> None:
        """
        Almacena una transicion.
        Para qMDP (use_qmdp=True) pasar tambien p0, p1, next_s_k0, next_s_k1.
        """
        self.buffer.push(
            state, action, reward, next_state, done,
            p0, p1, next_s_k0, next_s_k1,
        )

    # ------------------------------------------------------------------
    # Actualizacion — MDP (Ec. S17) o qMDP (Ec. S18)
    # ------------------------------------------------------------------

    def update(self) -> Optional[float]:
        """Un paso de gradiente. None si el buffer < min_buffer."""
        if len(self.buffer) < self.min_buffer:
            return None

        (states, actions, rewards, next_states,
         dones, qmdp_data) = self.buffer.sample(self.batch_size, self.device)

        q_values = self.q_online(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            if self.use_qmdp and qmdp_data is not None:
                # ── qMDP update (Ec. S18) ──────────────────────────
                p0   = qmdp_data["p0"]
                p1   = qmdp_data["p1"]
                s_k0 = qmdp_data["next_k0"]
                s_k1 = qmdp_data["next_k1"]

                # doble-Q por rama
                a_k0 = self.q_online(s_k0).argmax(dim=1, keepdim=True)
                a_k1 = self.q_online(s_k1).argmax(dim=1, keepdim=True)
                q_k0 = self.q_target(s_k0).gather(1, a_k0).squeeze(1)
                q_k1 = self.q_target(s_k1).gather(1, a_k1).squeeze(1)

                # terminal POR RAMA: si la rama ya es pura, su bootstrap es 0
                term0 = s_k0.max(dim=1).values > self.purity_threshold
                term1 = s_k1.max(dim=1).values > self.purity_threshold
                q_k0 = torch.where(term0, torch.zeros_like(q_k0), q_k0)
                q_k1 = torch.where(term1, torch.zeros_like(q_k1), q_k1)

                q_next = p0 * q_k0 + p1 * q_k1
                # sin (1-dones) global: el terminal ya se maneja por rama
                targets = rewards + self.gamma * q_next

            else:
                # ── MDP update estandar (Ec. S17) ─────────────────
                best_actions = self.q_online(next_states).argmax(
                    dim=1, keepdim=True
                )
                q_next = self.q_target(next_states).gather(
                    1, best_actions
                ).squeeze(1)
                targets = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Soft target update (Tabla S1: tau=0.001) ─────────────────
        self._update_steps += 1
        tau = self.tau_update
        if tau >= 1.0:
            self.q_target.load_state_dict(self.q_online.state_dict())
        else:
            for p_online, p_target in zip(
                self.q_online.parameters(),
                self.q_target.parameters()
            ):
                p_target.data.copy_(
                    tau * p_online.data + (1.0 - tau) * p_target.data
                )

        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "q_online":     self.q_online.state_dict(),
            "q_target":     self.q_target.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "eps":          self.eps,
            "n_episodes":   self._n_episodes,
            "update_steps": self._update_steps,
        }, path)

    def load(self, path: str) -> None:
        #ckpt = torch.load(path, map_location=self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_online.load_state_dict(ckpt["q_online"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.eps           = ckpt["eps"]
        self._n_episodes   = ckpt["n_episodes"]
        self._update_steps = ckpt.get("update_steps", 0)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "qMDP" if self.use_qmdp else "MDP"
        return (
            f"DQNAgent(NS={self.n_states}, NA={self.n_actions}, "
            f"mode={mode}, eps={self.eps:.4f}, "
            f"buffer={len(self.buffer)}/{self.buffer.buffer.maxlen})"
        )


# =====================================================================
# Test rapido
# =====================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("DQN Agent — test de integracion")
    print("=" * 55)

    NS = 16    # estados CaH+ J<=2
    NA = 13    # pulsos Tabla S2

    N_TRAINING = 600   # episodios (Fig. 2b del paper)

    agent = DQNAgent(
        n_states    = NS,
        n_actions   = NA,
        lr          = 5e-4,        # Tabla S1
        tau_update  = 0.001,       # Tabla S1
        eps_end     = 0.005,       # Sec. SD
        batch_size  = 32,          # Sec. SD
        N_training  = N_TRAINING,
        use_qmdp    = False,       # MDP para CaH+ J<=2
        loss_type   = 'mse',
    )
    print(f"\n{agent}")

    print(f"\nDecaimiento de epsilon (Ec. S16, tau_eps={agent.tau_eps:.0f}):")
    for n in [0, 50, 100, 180, 300, 500, 600]:
        print(f"  n={n:4d}:  eps={agent._compute_eps(n):.4f}")

    # Verificar que end_episode y decay_epsilon son el mismo metodo
    print(f"\nend_episode is decay_epsilon: "
          f"{DQNAgent.end_episode is DQNAgent.decay_epsilon}")

    print("\nSimulando transiciones...")
    rng = np.random.default_rng(0)
    for _ in range(600):
        s  = rng.dirichlet(np.ones(NS))
        a  = agent.select_action(s)
        r  = -1.0
        s2 = rng.dirichlet(np.ones(NS))
        done = rng.random() < 0.1
        agent.store(s, a, r, s2, done)

    loss = agent.update()
    print(f"Primer update: loss = {loss:.6f}" if loss else "Buffer insuficiente")

    # Decaimiento idempotente por indice de episodio
    for ep in range(1, 11):
        agent.decay_epsilon(ep)
    print(f"\nEpsilon tras 10 episodios (idempotente): {agent.eps:.4f}")
    print(f"  Esperado (Ec. S16, n=10):               {agent._compute_eps(10):.4f}")

    s_test = np.ones(NS) / NS
    print(f"\nAccion greedy en estado uniforme: "
          f"{agent.select_action(s_test, explore=False)}")

    print("\nTest completado.")