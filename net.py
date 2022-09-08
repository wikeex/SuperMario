import copy

import torch
from torch import nn
from torchvision import models


class BaseNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if torch.cuda.is_available():
            self.lstm_init1 = torch.randn(3, 256).cuda()
            self.lstm_init2 = torch.randn(3, 256).cuda()
        else:
            self.lstm_init1 = torch.randn(3, 256)
            self.lstm_init2 = torch.randn(3, 256)

        self.resnet18 = models.resnet18(input_dim, pretrained=True)
        self.resnet18_front = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=3)

        self.a_back_bone = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.v_back_bone = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, input, ignore_front=False):
        if ignore_front is not True:
            with torch.no_grad():
                front_t = self.resnet18_front(input)
                flatten_t = self.flatten(front_t)
        else:
            flatten_t = input

        # TODO: 直接输出resnet的结果作为state和next_state
        lstm_t, _ = self.lstm(flatten_t, (self.lstm_init1, self.lstm_init2))
        a_t: torch.Tensor = self.a_back_bone(lstm_t)
        v_t = self.v_back_bone(lstm_t)
        return a_t + v_t - torch.mean(a_t, dim=-1, keepdim=True), flatten_t


class MarioNet(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = BaseNet(input_dim, output_dim)
        self.target = BaseNet(input_dim, output_dim)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model, ignore_front=False):
        if model == "online":
            return self.online(input, ignore_front)
        elif model == "target":
            return self.target(input, ignore_front)
