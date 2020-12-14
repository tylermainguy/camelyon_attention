import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    """
    Class that takes in the hidden state from the last timestep (glimpse)
    and outputs the classification as a softmax layer (could also just
    do a single neuron I guess? bc binary classification)
    This is the same as the action network in the original paper.
    """

    def __init__(self, hidden_size):

        super().__init__()

        # binary prediction
        self.fc1 = nn.Linear(hidden_size, 2)

    def forward(self, h_t):

        # get log probabilities
        probabilities = F.log_softmax(self.fc1(h_t), dim=1)

        return probabilities
