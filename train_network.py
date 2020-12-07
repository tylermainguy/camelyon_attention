import torch
import torch.nn as nn
import torch.optim as optim

from recurrent_attention import RecurrentAttentionModel


def train(train_loader, batch_size):
    """
    Train the neural network on batches of data.
    """

    num_glimpses = 3

    # need to confirm these values
    glimpse_size = 299
    location_hidden_size = 128
    glimpse_feature_size = 128
    location_output_size = 2
    hidden_state_size = 256

    # create model
    model = RecurrentAttentionModel(
        glimpse_size=glimpse_size,
        location_hidden_size=location_hidden_size,
        glimpse_feature_size=glimpse_feature_size,
        location_output_size=location_output_size,
        hidden_state_size=hidden_state_size
    )

    # use gpu
    device = torch.device("cuda")

    # send model to gpu
    model.to(device)

    # not sure if I need this
    criterion = nn.BCELoss()

    # use adam optimizer for network
    optimizer = optim.Adam(model.parameters())

    # sample initial location from uniform distribution
    l_t = torch.FloatTensor(2, 1).uniform_(-1, 1).to(device)

    print(l_t.shape)
    h_t = torch.randn(
        1,
        batch_size,
        hidden_state_size,
        dtype=torch.float,
        device=device,
        requires_grad=True,
    )

    cell_state = torch.randn(
        1,
        batch_size,
        hidden_state_size,
        dtype=torch.float,
        device=device,
        requires_grad=True
    )

    for i, (x, y) in enumerate(train_loader, 0):

        x, y = x.to(device), y.to(device)

        locations = []
        for j in range(num_glimpses - 1):
            h_t, cell_state, l_t = model(
                x, l_t, h_t, cell_state, False)

            locations.append(l_t)

        h_t, cell_state, l_t, prediction = model(
            x, l_t, h_t, cell_state, is_pred=True)

        locations.append(l_t)
        print(locations)
        print("Predictions: {}".format(prediction))
        print("Label: {}".format(y))
        return
