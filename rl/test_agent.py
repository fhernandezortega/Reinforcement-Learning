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
import numpy as np

from env.rlqls_env import RLQLSEnv
from rl.dqn import DQN


# =========================
# Load environment
# =========================

env = RLQLSEnv()

# =========================
# Load model
# =========================

model = DQN(
    n_states=env.n_states,
    n_actions=env.n_actions,
)

model.load_state_dict(
    torch.load("dqn_model.pt")
)

model.eval()

# =========================
# Test agent
# =========================

state, _ = env.reset()

done = False

total_reward = 0

step = 0

print("\nINITIAL STATE:")
print(state)

while not done:

    with torch.no_grad():

        q_values = model(
            torch.FloatTensor(state)
        )

        action = torch.argmax(q_values).item()

    next_state, reward, done, _, info = env.step(action)

    print("\n====================")
    print(f"STEP {step}")
    print("====================")

    print("Action:", action)

    print("State before:")
    print(state)

    print("State after:")
    print(next_state)

    print("Reward:", reward)

    state = next_state

    total_reward += reward

    step += 1

print("\n====================")
print("FINAL RESULT")
print("====================")

print("Final state:")
print(state)

print("Total reward:", total_reward)