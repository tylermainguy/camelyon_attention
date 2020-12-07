import torch
import torch.nn as nn
import torch.nn.functional as F

from glimpse_network import GlimpseNetwork
from core_network import CoreNetwork
from classification_network import ClassificationNetwork


class RecurrentAttentionModel(nn.Module):
    """
    Implementation of the recurrent attention model for the classificaiton of
    histopathology images. Major changes include the introduction of an LSTM
    unit instead of a vanilla RNN, and a pretrained inception network for
    feature extraction from glimpses.
    """

    def __init__(self, glimpse_size, location_hidden, output_size, hidden_size, input_size):
        super().__init__()

        self.glimpse_net = GlimpseNetwork(
            glimpse_size, location_hidden, output_size)

        self.core_net = CoreNetwork(input_size, hidden_size, output_size)

        self.classification_net = ClassificationNetwork(hidden_size)
        # params here

    def forward(self, image, location, h_t_prev, is_pred=False):
        """
        Forward pass, combining these units together.
        """

        g_t = self.glimpse_net(image, location)

        h_t, l_t = self.core_net(h_t_prev, g_t)

        # only want to produce classification on last iteration
        if is_pred:
            prediction = self.classification_net(h_t)

        return h_t, l_t
