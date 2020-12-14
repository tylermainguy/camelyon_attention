import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineNetwork(nn.Module):
    """
    Used for baseline adjustment in the REINFORCE algorithm. Reduces the
    variance in the approximation of the gradient
    (see https://arxiv.org/abs/1406.6247).
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc1(h_t.detach())
        return b_t
