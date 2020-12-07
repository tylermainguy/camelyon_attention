import torch
import torch.nn as nn
import torch.optim as optim

from recurrent_attention import RecurrentAttentionModel


def train(train_loader):
    """
    Train the neural network on batches of data.
    """

    # need to confirm these values
    glimpse_size = 128
    location_hidden = 128
    output_size = 2
    hidden_size = 256
    input_size = 128

    # create model
    model = RecurrentAttentionModel(
        glimpse_size=glimpse_size,
        location_hidden=location_hidden,
        output_size=output_size,
        hidden_size=hidden_size,
        input_size=input_size
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
    init_l = torch.FloatTensor(2, 1).uniform_(-1, 1)

    init_h_t = t

    for i, (x, y) in enumerate(train_loader, 0):
        print("X shape: {}".format(x.shape))
        print("y shape: {}".format(y.shape))

        results = model(x)
    return 0


def main():
    train()


if __name__ == "__main__":
    main()
