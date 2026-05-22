import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(
        self,
        n_states,
        n_actions
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(n_states, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, n_actions),
        )

    def forward(self, x):

        return self.net(x)