import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2),
        )

    def forward(self, x):
        t = self.linear_relu_stack(x)
        t, _ = torch.max(t, 1)
        t = t.unsqueeze(1)
        return t
