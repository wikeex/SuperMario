import copy

import torch
from torch import nn


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        c, h, w = state_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.Sigmoid(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state, model):
        if model == "online":
            return self.online(state)
        elif model == "target":
            return self.target(state)


class BaseCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        c, h, w = state_dim

        self.back_bone = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.state = nn.Linear(3136, 256)

        self.action = nn.Linear(action_dim, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        back_bone_t = self.back_bone(state)
        state_t = self.state(back_bone_t)
        action_t = self.action(action)
        return self.q(torch.relu(state_t+action_t))


class CriticNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        c, h, w = state_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = BaseCritic(state_dim, action_dim)
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, state, action, model):
        if model == "online":
            return self.online(state, action)
        elif model == "target":
            return self.target(state, action)
