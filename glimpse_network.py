import torch
import torch.nn as nn
import torch.nn.functional as F

from glimpse_sensor import GlimpseSensor


class GlimpseNetwork(nn.Module):
    """
    Composing the general glimpse network. This will utilize a pretrained
    model (CNN) to compute a feature vector for a given glimpse, and a fully
    connected network to compute a feature vector for the location l. The
    two will be combined, and used to generate a glimpse network feature
    to be used to updated the hidden state.
    """

    def __init__(self, glimpse_size, location_hidden, output_size):
        super().__init__()

        self.glimpse_size = glimpse_size
        self.location_hidden = location_hidden

        # input to the location network is always size 2
        self.location_fc1 = nn.Linear(2, location_hidden)

        # size of this will be based on the output of the inception network
        # (or whatever other CNN network) convolutional layers look like

        self.location_fc2 = nn.Linear(location_hidden, output_size)
        # TODO add CNN layers here

    def forward(self, image, location):

        sensor = GlimpseSensor(self.glimpse_size)

        glimpse = sensor.glimpse(image, location)

        l_hidden = F.relu(self.location_fc1(location))
        l_out = self.location_fc2(l_hidden)

        # this should be replaced with CNN code
        glimpse_hidden = 0

        # replaced with fully connected layer
        glimpse_out = 0

        # TODO combine the output of the location and conv. layers
        glimpse_feature = F.relu(l_out + glimpse_out)
