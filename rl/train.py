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

from env.rlqls_env import RLQLSEnv
from rl.dqn import DQN
from rl.replay_buffer import ReplayBuffer


# =========================
# Environment
# =========================

env = RLQLSEnv()

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
    lr=5e-4,
)

loss_fn = nn.MSELoss()

# =========================
# Replay Buffer
# =========================

buffer = ReplayBuffer(
    capacity=10000
)

# =========================
# Hyperparameters
# =========================

episodes = 1000

batch_size = 32

gamma = 0.99

epsilon = 1.0

epsilon_min = 0.005

epsilon_decay = 0.998

target_update = 10

# =========================
# Training
# =========================

for episode in range(episodes):

    state, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        # =====================
        # epsilon-greedy
        # =====================

        if np.random.rand() < epsilon:

            action = env.action_space.sample()

        else:

            with torch.no_grad():

                q_values = model(
                    torch.FloatTensor(state)
                )

                action = torch.argmax(
                    q_values
                ).item()

        # =====================
        # environment step
        # =====================

        next_state, reward, done, _, info = env.step(action)

        total_reward += reward

        # =====================
        # save experience
        # =====================

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        # =====================
        # train qMDP
        # =====================

        if len(buffer) >= batch_size:

            (
                states,
                actions,
                rewards,
                next_states,
                dones,
            ) = buffer.sample(batch_size)

            states_t      = torch.FloatTensor(states)
            actions_t     = torch.LongTensor(actions)
            rewards_t     = torch.FloatTensor(rewards)
            dones_t       = torch.FloatTensor(dones)

            # =================
            # current Q
            # =================

            current_q = model(states_t)

            current_q = current_q.gather(
                1,
                actions_t.unsqueeze(1)
            ).squeeze(1)

            # =================
            # qMDP target Q
            # (Eq. S18 of paper)
            # =================

            with torch.no_grad():

                target_q_vals = torch.zeros(batch_size)

                for idx in range(batch_size):

                    s   = states[idx]
                    a   = int(actions[idx])
                    d   = dones[idx]

                    if d:
                        target_q_vals[idx] = rewards_t[idx]
                        continue

                    # get transition matrices
                    A0, A1 = env.get_transition_matrices(a)

                    # post-measurement states
                    s0 = A0 @ s
                    s1 = A1 @ s

                    p0 = float(np.sum(s0))
                    p1 = float(np.sum(s1))

                    total = p0 + p1 + 1e-12
                    p0 /= total
                    p1 /= total

                    # normalize post-measurement states
                    s0_norm = s0 / (np.sum(s0) + 1e-12)
                    s1_norm = s1 / (np.sum(s1) + 1e-12)

                    s0_t = torch.FloatTensor(
                        s0_norm.astype(np.float32)
                    )
                    s1_t = torch.FloatTensor(
                        s1_norm.astype(np.float32)
                    )

                    # qMDP update: weighted by p0, p1
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
                        * (1 - dones_t[idx])
                    )

            # =================
            # loss
            # =================

            loss = loss_fn(
                current_q,
                target_q_vals
            )

            # =================
            # backpropagation
            # =================

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    # =========================
    # epsilon decay
    # =========================

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

    # =========================
    # target network update
    # =========================

    if episode % target_update == 0:

        target_model.load_state_dict(
            model.state_dict()
        )

    # =========================
    # print
    # =========================

    print(
        f"Episode {episode}, "
        f"reward={total_reward:.2f}, "
        f"epsilon={epsilon:.3f}",
        flush=True
    )

torch.save(model.state_dict(), "dqn_model.pt")