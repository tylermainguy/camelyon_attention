import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_network import BaselineNetwork
from classification_network import ClassificationNetwork
from core_network import CoreNetwork
from glimpse_network import GlimpseNetwork
from location_network import LocationNetwork


class RecurrentAttentionModel(nn.Module):
    """
    Implementation of the recurrent attention model for the classificaiton of
    histopathology images. Major changes include the introduction of an LSTM
    unit instead of a vanilla RNN, and a pretrained inception network for
    feature extraction from glimpses.
    """

    def __init__(self, glimpse_size, location_hidden_size, glimpse_feature_size, hidden_state_size, location_output_size, std, glimpse_hidden, channels, num_patches, zoom_amt):
        super().__init__()

        self.glimpse_net = GlimpseNetwork(
            glimpse_size, location_hidden_size, glimpse_feature_size, glimpse_hidden, channels, num_patches, zoom_amt)

        self.core_net = CoreNetwork(
            glimpse_feature_size, hidden_state_size, location_output_size)

        self.location_net = LocationNetwork(hidden_state_size, 2, std)

        self.classification_net = ClassificationNetwork(hidden_state_size)

        self.baseline_net = BaselineNetwork(hidden_state_size, 1)
        # params here

    def forward(self, image, location, h_t_prev, is_pred=False):
        """
        Forward pass, combining these units together.
        """

        g_t = self.glimpse_net(image, location)
        print("GLIMPSE SIZE: {}".format(g_t.shape))
        h_t = self.core_net(g_t, h_t_prev)
        print("CORE NET SHAPE: {}".format(h_t.shape))
        # # remove "len_seq" from lstm_out
        # h_t = torch.squeeze(h_t_prev, 1)

        baseline = self.baseline_net(h_t).squeeze()

        log_pi, l_t = self.location_net(h_t)

        # only want to produce classification on last iteration
        if is_pred:
            prediction = self.classification_net(h_t)
            return l_t, h_t, log_pi, baseline, prediction

        return l_t, h_t, log_pi, baseline
