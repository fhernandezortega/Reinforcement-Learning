import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import torch
import torch.nn as nn
import numpy as np
import json

from env.rlqls_env_cah import RLQLSEnvCaH
from rl.dqn import DQN
from rl.replay_buffer import ReplayBuffer


# =========================
# Reproducibility
# =========================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# Environment
# =========================

env = RLQLSEnvCaH(
    T=300.0,
    purity_threshold=0.99
)

print(
    f"States: {env.n_states}, "
    f"Actions: {env.n_actions}",
    flush=True
)

# =========================
# Checkpoints directory
# =========================

os.makedirs("checkpoints", exist_ok=True)

# =========================
# Networks
# =========================

model = DQN(
    n_states=env.n_states,
    n_actions=env.n_actions,
)

target_model = DQN(
    n_states=env.n_states,
    n_actions=env.n_actions,
)

target_model.load_state_dict(
    model.state_dict()
)

# =========================
# Optimizer
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4
)

loss_fn = nn.SmoothL1Loss()

# =========================
# Replay Buffer
# =========================

buffer = ReplayBuffer(
    capacity=10000
)

# =========================
# Hyperparameters (Sec. SD)
# =========================

episodes      = 1000
batch_size    = 32
gamma         = 1.0

epsilon_start = 1.0
epsilon_end   = 0.005
tau_epsilon   = 0.3 * episodes

target_tau    = 0.001

# =========================
# Training statistics
# =========================

reward_history = []
step_history   = []
purity_history = []

# =========================
# Training
# =========================

for episode in range(episodes):

    state, _ = env.reset()

    done = False

    total_reward = 0.0

    step_count = 0

    while not done:

        # =====================
        # epsilon (Eq. S16)
        # =====================

        epsilon = epsilon_end + (
            epsilon_start - epsilon_end
        ) * np.exp(-episode / tau_epsilon)

        # =====================
        # epsilon-greedy
        # =====================

        if np.random.rand() < epsilon:

            action = env.action_space.sample()

        else:

            with torch.no_grad():

                state_t = torch.FloatTensor(
                    state
                )

                q_values = model(state_t)

                action = torch.argmax(
                    q_values
                ).item()

        # =====================
        # environment step
        # =====================

        next_state, reward, done, _, info = (
            env.step(action)
        )

        total_reward += reward

        step_count += 1

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        # =====================
        # Learn
        # =====================

        if len(buffer) >= batch_size:

            (
                states,
                actions,
                rewards,
                next_states,
                dones,
            ) = buffer.sample(batch_size)

            states_t  = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            dones_t   = torch.FloatTensor(dones)

            current_q = model(states_t)

            current_q = current_q.gather(
                1,
                actions_t.unsqueeze(1)
            ).squeeze(1)

            with torch.no_grad():

                target_q_vals = torch.zeros(
                    batch_size
                )

                for idx in range(batch_size):

                    s = states[idx]
                    a = int(actions[idx])

                    if dones[idx]:

                        target_q_vals[idx] = (
                            rewards_t[idx]
                        )

                        continue

                    A0, A1 = (
                        env.get_transition_matrices(a)
                    )

                    s0 = A0 @ s
                    s1 = A1 @ s

                    p0 = float(np.sum(s0))
                    p1 = float(np.sum(s1))

                    total = p0 + p1 + 1e-12

                    p0 /= total
                    p1 /= total

                    if np.sum(s0) < 1e-10:
                        s0_norm = (
                            np.ones_like(s0) / len(s0)
                        )
                    else:
                        s0_norm = s0 / (
                            np.sum(s0) + 1e-12
                        )

                    if np.sum(s1) < 1e-10:
                        s1_norm = (
                            np.ones_like(s1) / len(s1)
                        )
                    else:
                        s1_norm = s1 / (
                            np.sum(s1) + 1e-12
                        )

                    s0_t = torch.FloatTensor(
                        s0_norm.astype(np.float32)
                    )

                    s1_t = torch.FloatTensor(
                        s1_norm.astype(np.float32)
                    )

                    q0 = torch.max(
                        target_model(s0_t)
                    ).item()

                    q1 = torch.max(
                        target_model(s1_t)
                    ).item()

                    qmdp_next = p0 * q0 + p1 * q1

                    target_q_vals[idx] = (
                        rewards_t[idx]
                        + gamma * qmdp_next
                    )

            loss = loss_fn(
                current_q,
                target_q_vals
            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0
            )

            optimizer.step()

            # soft target update
            for param, target_param in zip(
                model.parameters(),
                target_model.parameters()
            ):
                target_param.data.copy_(
                    target_tau * param.data
                    + (1 - target_tau)
                    * target_param.data
                )

    # =========================
    # Logging
    # =========================

    reward_history.append(total_reward)
    step_history.append(step_count)
    purity_history.append(info["purity"])

    avg_reward = np.mean(reward_history[-100:])
    avg_steps  = np.mean(step_history[-100:])
    avg_purity = np.mean(purity_history[-100:])

    print(
        f"Episode {episode} | "
        f"steps={step_count} | "
        f"reward={total_reward:.2f} | "
        f"purity={info['purity']:.3f} | "
        f"avg_steps={avg_steps:.2f} | "
        f"avg_purity={avg_purity:.3f} | "
        f"epsilon={epsilon:.3f}",
        flush=True
    )

    # =========================
    # Save checkpoint every 50
    # episodes (for Fig. 2c)
    # =========================

    if (episode + 1) % 50 == 0:

        torch.save(
            model.state_dict(),
            f"checkpoints/model_ep{episode+1}.pt"
        )

        print(
            f"Checkpoint guardado: "
            f"checkpoints/model_ep{episode+1}.pt",
            flush=True
        )

# =========================
# Save final model
# =========================

torch.save(
    model.state_dict(),
    "dqn_cah_model.pt"
)

# =========================
# Save training history
# =========================

history = {
    "steps": step_history,
    "rewards": reward_history,
    "purity": purity_history,
}

with open("training_history.json", "w") as f:
    json.dump(history, f)

print("Model saved.", flush=True)
print("Historia guardada.", flush=True)