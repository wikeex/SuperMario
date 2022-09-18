import copy

import torch
from torch import nn


class BaseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.back_bone = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.front_bone = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=24, stride=10)
        )

    def forward(self, input, is_training=False):
        t = self.back_bone(input)
        if not is_training:
            return t

        front_t = self.front_bone(t)
        t, _ = torch.max(front_t, 0)
        t = t.unsqueeze(0)
        return t


class MarioNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.vision_bone = BaseNet(input_dim)

        self.online = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        vision_t = self.vision_bone(input)
        if model == "online":
                return self.online(vision_t)
        elif model == "target":
            return self.target(vision_t)
