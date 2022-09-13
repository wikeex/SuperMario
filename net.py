import copy

import torch
from torch import nn
from vit_pytorch import ViT


class BaseNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.front_bone = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=2, end_dim=-1)
        )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=49, nhead=7, dropout=0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.a_front = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3136, output_dim)
        )

    def forward(self, input):
        front_t = self.front_bone(input)
        transformer_encoder_t = self.transformer_encoder(front_t)
        a_t: torch.Tensor = self.a_front(transformer_encoder_t)
        return a_t


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

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
