import copy

import torch
from torch import nn
from torchvision import models


class BaseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        c, h, w = input_dim
        self.resnet18 = models.resnet18()
        self.back_bone = nn.Sequential(
            *list(self.resnet18.children())[:-2]
        )
        self.front_bone = nn.Sequential(
            nn.ConvTranspose2d(512, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, input, is_training=False):
        t = self.back_bone(input)
        if not is_training:
            return t

        front_t = self.front_bone(t)
        return t, front_t


class MarioNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):

        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
