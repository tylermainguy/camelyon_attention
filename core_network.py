import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreNetwork(nn.Module):
    """
    Recurrent unit of the attention mechanism. This will consist of LSTM units
    that will be used to generate a new recurrent hidden state at each
    timestep (glimpse). This hidden state will be used by the location network
    to decide the next area to glimpse.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        hidden size is the size of the hidden state in the recurrent unit.
        input size is the size of the sequence data (in this case, the
        glimpse feature vector)
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # lstm network
        self.lstm = nn.LSTM(input_size, hidden_size)

        # fully connected layer to predict the next location
        self.location_fc = nn.Linear(hidden_size, output_size)

    def forward(self, h_t_prev, glimpse_feature):
        """
        forward pass of the recurrent unit. output is the updated hidden
        state of the hidden unit
        """

        # pass in new feature and previous hidden state into LSTM
        h_t = self.lstm(glimpse_feature, h_t_prev)

        # get next location vector
        l_t = self.location(h_t)

        return h_t, l_t
