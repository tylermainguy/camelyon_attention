import torch
import torch.nn as nn
import torch.nn.functional as F

from glimpse_sensor import GlimpseSensor
from inception_pretrained import get_pretrained_inception


class GlimpseNetwork(nn.Module):
    """
    Composing the general glimpse network. This will utilize a pretrained
    model (CNN) to compute a feature vector for a given glimpse, and a fully
    connected network to compute a feature vector for the location l. The
    two will be combined, and used to generate a glimpse network feature
    to be used to updated the hidden state.
    """

    def __init__(self, glimpse_size, location_hidden_size, glimpse_feature_size, glimpse_hidden, channels, num_patches, zoom_amt):
        super().__init__()

        self.glimpse_size = glimpse_size
        self.location_hidden_size = location_hidden_size
        self.glimpse_hidden = glimpse_hidden
        self.channels = channels
        self.num_patches = num_patches

        self.sensor = GlimpseSensor(
            glimpse_size, num_zooms=num_patches, zoom_amt=zoom_amt)

        # input size is flattened glimpse
        in_dim = channels * num_patches * glimpse_size * glimpse_size

        # for CNN based implementation
        # self.inception = get_pretrained_inception(glimpse_feature_size)

        # glimpse fully connected network
        self.glimpse1 = nn.Linear(in_dim, glimpse_hidden)
        self.glimpse2 = nn.Linear(glimpse_hidden, glimpse_feature_size)

        # input to the location network is always size 2
        self.location_fc1 = nn.Linear(2, location_hidden_size)
        self.location_fc2 = nn.Linear(
            location_hidden_size, glimpse_feature_size)

    def forward(self, image, location):

        # get glimpse from "sensor"
        glimpse = self.sensor.glimpse(image, location)

        # flatten location, pass through fully connected layers
        location = location.view(location.size(0), -1)
        l_hidden = F.relu(self.location_fc1(location))
        l_out = self.location_fc2(l_hidden)

        # pass glimpse through fully connected layers
        glimpse = F.relu(self.glimpse1(glimpse))
        glimpse_out = self.glimpse2(glimpse)

        # combination of feature vectors, passed through activation
        glimpse_feature = F.relu(l_out + glimpse_out)

        return glimpse_feature
