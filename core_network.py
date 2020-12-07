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

    def __init__(self, glimpse_feature_size, hidden_state_size, location_output_size):
        """
        hidden size is the size of the hidden state in the recurrent unit.
        input size is the size of the sequence data (in this case, the
        glimpse feature vector)
        """

        super().__init__()
        self.hidden_size = hidden_state_size
        self.input_size = glimpse_feature_size

        # lstm network
        self.lstm = nn.LSTM(glimpse_feature_size, hidden_state_size)

        # fully connected layer to predict the next location
        self.location_fc = nn.Linear(hidden_state_size, location_output_size)

    def forward(self, h_t_prev, cell_state, glimpse_feature):
        """
        forward pass of the recurrent unit. output is the updated hidden
        state of the hidden unit
        """
        # need to add number of timesteps per input (only one timestep)
        glimpse_feature = glimpse_feature.unsqueeze(0)

        # pass in new feature and previous hidden state into LSTM
        lstm_out, (h_t, cell_state) = self.lstm(
            glimpse_feature, (h_t_prev, cell_state))

        # get next location vector
        l_t = self.location_fc(lstm_out)

        l_t = torch.squeeze(l_t, 0)
        l_t = torch.transpose(l_t, 0, 1)
        print(l_t.shape)
        return h_t, cell_state, l_t
