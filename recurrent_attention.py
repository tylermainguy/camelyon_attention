import torch
import torch.nn as nn
import torch.nn.functional as F

from glimpse_network import GlimpseNetwork
from core_network import CoreNetwork
from classification_network import ClassificationNetwork
from location_network import LocationNetwork
from baseline_network import BaselineNetwork


class RecurrentAttentionModel(nn.Module):
    """
    Implementation of the recurrent attention model for the classificaiton of
    histopathology images. Major changes include the introduction of an LSTM
    unit instead of a vanilla RNN, and a pretrained inception network for
    feature extraction from glimpses.
    """

    def __init__(self, glimpse_size, location_hidden_size, glimpse_feature_size, hidden_state_size, location_output_size, std):
        super().__init__()

        self.glimpse_net = GlimpseNetwork(
            glimpse_size, location_hidden_size, glimpse_feature_size)

        self.core_net = CoreNetwork(
            glimpse_feature_size, hidden_state_size, location_output_size)

        self.location_net = LocationNetwork(hidden_state_size, 2, std)

        self.classification_net = ClassificationNetwork(hidden_state_size)

        self.baseline_net = BaselineNetwork(hidden_state_size, 1)
        # params here

    def forward(self, image, location, h_t_prev, cell_state, is_pred=False):
        """
        Forward pass, combining these units together.
        """

        g_t = self.glimpse_net(image, location)

        lstm_out, h_t, cell_state = self.core_net(h_t_prev, cell_state, g_t)

        # remove "len_seq" from lstm_out
        lstm_out = torch.squeeze(lstm_out, 1)

        baseline = self.baseline_net(lstm_out).squeeze()

        log_pi, l_t = self.location_net(lstm_out)

        # only want to produce classification on last iteration
        if is_pred:
            prediction = self.classification_net(lstm_out)
            return h_t, cell_state, l_t, log_pi, baseline, prediction

        return h_t, cell_state, l_t, log_pi, baseline
