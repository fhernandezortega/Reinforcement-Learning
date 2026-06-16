"""
DQN Agent — RL-QLS (PIPI2026 Sec. SD)
=======================================
Deep Q-Network con double-Q networks y experience replay para
preparacion de estados moleculares puros.

Diferencias clave respecto a un DQN estandar:
  - gamma = 1.0 exactamente (tarea episodica sin descuento, Sec. SD)
  - Decaimiento de epsilon segun Ec. S16 del paper:
        eps(n) = eps_end + (eps_start - eps_end) * exp(-n / tau_eps)
    con tau_eps = 0.3 * N_training
  - Actualizacion suave del target network con parametro tau (soft update)
    O actualizacion dura cada C pasos (configurable)
  - Soporte para qMDP temporal difference update (Ec. S18)
    que incorpora explicitamente las probabilidades de medicion POVM
  - Funcion de perdida: MSE o Smooth L1 (ambas reportadas en Sec. SD)
  - Red neuronal: 3 capas ocultas de 128 nodos, ReLU (Sec. SD)
    Tambien se prueba 4 capas (con peor rendimiento segun el paper)

Hiperparametros reportados para CaH+ J in {1,2} (Tabla S1 / Sec. SD):
  tau_update = 0.001, lr = 0.0005, eps_end = 0.005

Referencia: PIPI2026 Sec. SD, Ec. S14-S18
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Optional, Tuple, List
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

    El paper tambien probo 4 capas y redes mas anchas con los
    mismos resultados cualitativos.
    """

    def __init__(
        self,
        n_states:    int,
        n_actions:   int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        # Por defecto: 3 capas de 128 nodos (Sec. SD)
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

    Almacena transiciones (s, a, r, s', done) y muestrea
    mini-batches aleatorios para el entrenamiento.

    Para qMDP tambien almacena (p0, p1, s'_k0, s'_k1) que permiten
    el update de Ec. S18 sin re-ejecutar el simulador.
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
        # Campos adicionales para qMDP (opcionales)
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
        """
        Muestrea un mini-batch.

        Returns
        -------
        states, actions, rewards, next_states, dones : tensores
        qmdp_data : dict con p0, p1, next_s_k0, next_s_k1 (o None si MDP)
        """
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

        # Datos qMDP (None si no se almacenaron)
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

    Implementa exactamente los algoritmos descritos en PIPI2026 Sec. SD:

    1. Exploracion epsilon-greedy con decaimiento exponencial (Ec. S16)
    2. Double-Q: accion elegida por red online, valor de red target
    3. Soft target update: theta_target = tau*theta_online + (1-tau)*theta_target
       (o hard update si tau_update = 1.0)
    4. qMDP temporal difference (Ec. S18) cuando use_qmdp=True

    Parameters (hiperparametros del paper)
    ----------------------------------------
    n_states      : NS = 16 para J<=2
    n_actions     : NA = 13 (Tabla S2) o 36 (auto-enumeracion)
    hidden_dims   : [128, 128, 128]  (Sec. SD)
    lr            : 0.0005  (Tabla S1, mejor para CaH+ J<=2)
    tau_update    : 0.001   (soft update, Tabla S1)
    eps_start     : 1.0     (Ec. S16)
    eps_end       : 0.005   (Sec. SD, mejor resultado)
    N_training    : numero total de episodios de entrenamiento
                    tau_eps = 0.3 * N_training  (Ec. S16)
    gamma         : 1.0     (sin descuento, Sec. SD)
    loss_type     : 'mse' o 'smooth_l1' (ambas en Sec. SD)
    use_qmdp      : True para qMDP update (Ec. S18), False para MDP (Ec. S17)
    """

    def __init__(
        self,
        n_states:       int,
        n_actions:      int,
        hidden_dims:    Optional[List[int]] = None,
        # Replay buffer
        buffer_capacity: int   = 50_000,
        batch_size:      int   = 64,
        min_buffer:      int   = 512,
        # Optimizacion
        lr:              float = 5e-4,       # Tabla S1: rl=0.0005
        gamma:           float = 1.0,        # sin descuento (Sec. SD)
        loss_type:       str   = 'mse',      # 'mse' o 'smooth_l1'
        # Exploracion — Ec. S16
        eps_start:       float = 1.0,
        eps_end:         float = 0.005,      # mejor resultado Sec. SD
        N_training:      int   = 1000,       # episodios totales de entrenamiento
        # Target network
        tau_update:      float = 0.001,      # Tabla S1: tau=0.001 (soft update)
        # qMDP
        use_qmdp:        bool  = False,
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
        # eps(n) = eps_end + (eps_start - eps_end) * exp(-n / tau_eps)
        # tau_eps = 0.3 * N_training
        self.eps_start   = eps_start
        self.eps_end     = eps_end
        self.tau_eps     = 0.3 * N_training
        self._n_episodes = 0       # contador de episodios para Ec. S16
        self.eps         = eps_start

        self._update_steps = 0

    # ------------------------------------------------------------------
    # Exploracion — Ec. S16
    # ------------------------------------------------------------------

    def _compute_eps(self, n: int) -> float:
        """
        Ec. S16: eps(n) = eps_end + (eps_start - eps_end) * exp(-n / tau_eps)
        """
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-n / max(self.tau_eps, 1e-8))

    def decay_epsilon(self) -> float:
        """
        Llama una vez al final de cada episodio de entrenamiento.
        Actualiza eps segun Ec. S16 con el contador de episodios.
        """
        self._n_episodes += 1
        self.eps = self._compute_eps(self._n_episodes)
        return self.eps

    # ------------------------------------------------------------------
    # Seleccion de accion
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, explore: bool = True) -> int:
        """
        Seleccion epsilon-greedy.

        Durante entrenamiento: explore=True  (usa self.eps actual)
        Durante evaluacion:    explore=False (greedy puro)
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

        Para uso con qMDP (use_qmdp=True), pasar tambien:
          p0, p1       : probabilidades de medicion k=0, k=1
          next_s_k0/k1 : estados post-medicion segun cada rama
        """
        self.buffer.push(
            state, action, reward, next_state, done,
            p0, p1, next_s_k0, next_s_k1,
        )

    # ------------------------------------------------------------------
    # Actualizacion — MDP (Ec. S17) o qMDP (Ec. S18)
    # ------------------------------------------------------------------

    def update(self) -> Optional[float]:
        """
        Un paso de gradiente. Devuelve la perdida o None si el buffer
        tiene menos muestras que min_buffer.

        MDP update (Ec. S17):
            delta Q(s,a) ∝ max_a' Q(s',a') + R - Q(s,a)

        qMDP update (Ec. S18):
            delta Q(s,a) ∝ p0*max_a' Q(s'_k0, a') +
                           p1*max_a' Q(s'_k1, a') + R - Q(s,a)
        """
        if len(self.buffer) < self.min_buffer:
            return None

        (states, actions, rewards, next_states,
         dones, qmdp_data) = self.buffer.sample(self.batch_size, self.device)

        # Q(s, a) — red online
        q_values = self.q_online(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            if self.use_qmdp and qmdp_data is not None:
                # ── qMDP update (Ec. S18) ──────────────────────────
                p0 = qmdp_data["p0"]          # (B,)
                p1 = qmdp_data["p1"]          # (B,)
                s_k0 = qmdp_data["next_k0"]   # (B, NS)
                s_k1 = qmdp_data["next_k1"]   # (B, NS)

                # Accion optima en cada rama segun red online (double-Q)
                a_k0 = self.q_online(s_k0).argmax(dim=1, keepdim=True)
                a_k1 = self.q_online(s_k1).argmax(dim=1, keepdim=True)

                # Valor segun red target
                q_k0 = self.q_target(s_k0).gather(1, a_k0).squeeze(1)
                q_k1 = self.q_target(s_k1).gather(1, a_k1).squeeze(1)

                # Ec. S18: valor esperado ponderado por probabilidades
                q_next = p0 * q_k0 + p1 * q_k1

            else:
                # ── MDP update estandar (Ec. S17) ─────────────────
                # Double-Q: accion de online, valor de target
                best_actions = self.q_online(next_states).argmax(
                    dim=1, keepdim=True
                )
                q_next = self.q_target(next_states).gather(
                    1, best_actions
                ).squeeze(1)

            # Target: R + gamma * Q_next * (1 - done)
            # gamma=1 y done=True en el paso terminal
            targets = rewards + self.gamma * q_next * (1.0 - dones)

        # Perdida y backprop
        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Clip de gradiente para estabilidad
        nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Soft target update ────────────────────────────────────────
        # theta_target = tau*theta_online + (1-tau)*theta_target
        # (Tabla S1: tau = 0.001)
        self._update_steps += 1
        tau = self.tau_update
        if tau >= 1.0:
            # Hard update: copiar pesos directamente
            self.q_target.load_state_dict(self.q_online.state_dict())
        else:
            # Soft update
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
            "q_online":    self.q_online.state_dict(),
            "q_target":    self.q_target.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "eps":         self.eps,
            "n_episodes":  self._n_episodes,
            "update_steps": self._update_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_online.load_state_dict(ckpt["q_online"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.eps            = ckpt["eps"]
        self._n_episodes    = ckpt["n_episodes"]
        self._update_steps  = ckpt.get("update_steps", 0)

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
    NA = 36    # pulsos auto-enumerados

    N_TRAINING = 600   # episodios (Fig. 2b del paper)

    agent = DQNAgent(
        n_states    = NS,
        n_actions   = NA,
        lr          = 5e-4,        # Tabla S1
        tau_update  = 0.001,       # Tabla S1
        eps_end     = 0.005,       # Sec. SD
        N_training  = N_TRAINING,
        use_qmdp    = False,       # MDP para CaH+ J<=2
        loss_type   = 'mse',
    )
    print(f"\n{agent}")

    # Verificar decaimiento de epsilon segun Ec. S16
    print(f"\nDecaimiento de epsilon (Ec. S16, tau_eps={agent.tau_eps:.0f}):")
    checkpoints = [0, 50, 100, 180, 300, 500, 600]
    for n in checkpoints:
        eps_n = agent._compute_eps(n)
        print(f"  n={n:4d}:  eps={eps_n:.4f}")

    # Simular algunas transiciones y un paso de entrenamiento
    print("\nSimulando transiciones...")
    rng = np.random.default_rng(0)
    for _ in range(600):   # llenar buffer hasta min_buffer
        s  = rng.dirichlet(np.ones(NS))     # estado aleatorio normalizado
        a  = agent.select_action(s)
        r  = -1.0                            # recompensa constante (Sec. SD)
        s2 = rng.dirichlet(np.ones(NS))
        done = rng.random() < 0.1
        agent.store(s, a, r, s2, done)

    loss = agent.update()
    print(f"Primer update: loss = {loss:.6f}" if loss else "Buffer insuficiente")

    # Simular decaimiento de epsilon por episodios
    for _ in range(10):
        agent.decay_epsilon()
    print(f"\nEpsilon despues de 10 episodios: {agent.eps:.4f}")
    print(f"  Esperado (Ec. S16): {agent._compute_eps(10):.4f}")

    # Seleccion greedy (evaluacion)
    s_test = np.ones(NS) / NS   # estado uniforme
    a_greedy = agent.select_action(s_test, explore=False)
    print(f"\nAccion greedy en estado uniforme: {a_greedy}")

    print("\nTest completado.")