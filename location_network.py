import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationNetwork(nn.Module):
    """
    Used to generate the next location to take a glimpse at given
    the output of the LSTM recurrent unit.
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.hid_size = input_size // 2

        self.std = std
        self.fc = nn.Linear(input_size, self.hid_size)
        self.fc_lt = nn.Linear(self.hid_size, output_size)

    def forward(self, h_t):
        # compute mean
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))

        # reparametrization trick
        l_t = torch.distributions.Normal(mu, self.std).rsample()
        l_t = l_t.detach()

        log_pi = torch.distributions.Normal(mu, self.std).log_prob(l_t)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        # bound between [-1, 1]
        l_t = torch.clamp(l_t, -1, 1)

        return log_pi, l_t
