The project is organized as follows:
Reinforcement Learning/
├── physics/
│   ├── hamiltonian_cah.py
│   ├── pulses_cah.py
│   └── hamiltonian.py
├── env/
│   └── rlqls_env_cah.py
├── rl/
│   ├── dqn.py
│   ├── replay_buffer.py
│   ├── train_cah.py
│   └── test_cah.py
├── plots/
│   ├── fig2a.py
│   ├── fig2b.py
│   └── fig2c.py
└── checkpoints/

Where
----physics/hamiltonian_cah.py 
Construct the molecular Hamiltonian for CaH⁺ according to Equation S9 in the paper:
H = 2π R J² - g μN J·B - gI μN I·B - 2π cIJ I·J

Using the NIST experimental parameters (B = 0.36 mT). Generate the 16 basis states |J, mJ, mI⟩ for the manifolds J=1 and J=2, and calculate the Boltzmann distribution at temperature T.
Output: 16×16 matrix H, energies, initial thermal distribution.

----physics/pulses_cah.py 
Define the library of 13 blue-sideband Raman pulses (Table S2 of the paper). For each pulse, calculate the transition matrices A₀ and A₁ according to Equations 4a–4b:

A₀: population remaining in the motonic state k=0
A₁: population transferred to the motonic state k=1

Output: a list of 13 precomputed (A₀, A₁) pairs.

-----env/rlqls_env_cah.py
A Gymnasium environment that implements the MDP of the RL-QLS protocol (Fig. 1c of the paper). At each step:

Apply the pulse selected by the agent
Calculate the measurement probabilities p₀ and p₁
Sample the result k ∈ {0,1}
Collapse the quantum state according to Eq. 4b
Calculates the reward: R = -1 per step, with an additional penalty if the state does not change
Terminates when purity = max(state) > 0.99 or max_steps is reached.

----rl/dqn.py
It contains two classes:
QNetwork — a fully connected neural network with 3 hidden layers of 128 nodes (as in the paper, Sec. SD).
DQNAgent — a complete DQN agent with:

Double-Q networks (online + target) to reduce overestimation
Experience replay with ReplayBuffer
Epsilon-greedy with exponential decay (Eq. S16)
Soft target update with parameter τ
qMDP update (Eq. S18): the target weights the two possible observation outcomes, k=0 and k=1, by their probabilities p₀ and p₁

----- rl/replay_buffer.py
A circular buffer that stores transitions (s, a, r, s', done). It allows for sampling random mini-batches for off-policy training.

----rl/train_cah.py
Main training loop. For each episode:

Reset the environment with a random sample from the manifold J=1
The agent selects actions using ε-greedy
Save transitions to the buffer
Call agent.update(env=env), which performs the qMDP update
Save checkpoints every 50 episodes to reproduce Fig. 2c

Hyperparameters from the paper: lr=5e-4, γ=1, τ=0.001, εend=0.005, batch=32.

-----rl/test_cah.py
Evaluate the trained model on 100 episodes with exploration disabled. Report the success rate, mean steps, and distribution of prepared terminal states.

---plots/ 
Scripts to reproduce the figures in the paper:

fig2a.py — CaH⁺ energy level diagram with arrows indicating the 13 pulses
fig2b.py — Training curve: individual steps (orange) + 100-episode moving average (blue) + sweeping protocol line
fig2c.py — Testing curve: mean steps vs. training episodes, evaluating each checkpoint

How They're Connected

hamiltonian_cah.py
       │
       ▼
pulses_cah.py  ──────────────────────┐
       │                             │
       ▼                             ▼
rlqls_env_cah.py          (A0_list, A1_list)
       │                             │
       ▼                             ▼
train_cah.py  ◄──── dqn.py (DQNAgent.update(env))
       │
       ▼
checkpoints/ ──► fig2c.py
       │
training_history.json ──► fig2b.py
