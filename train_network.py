import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from recurrent_attention import RecurrentAttentionModel


def train(train_loader, model, writer, epoch):
    """
    Train the neural network on batches of data.
    """

    num_glimpses = 3

    # need to confirm these values
    std = 0.05
    glimpse_size = 512
    location_hidden_size = 128
    glimpse_feature_size = 128
    location_output_size = 2
    hidden_state_size = 256

    # use gpu
    device = torch.device("cuda")

    loss_sum = 0
    # use adam optimizer for network
    optimizer = optim.Adam(model.parameters())

    for i, (x, y) in enumerate(train_loader):
        print("BATCH #{}".format(i))
        batch_size = x.shape[0]
        # sample initial location from uniform distribution
        l_t = torch.FloatTensor(batch_size, 2).uniform_(-1, 1).to(device)

        # need to add sequence length for LSTM
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

        # zero model gradients before starting pass
        optimizer.zero_grad()

        # send to GPU
        x, y = x.to(device), y.to(device)

        locations = []
        log_pis = []
        baselines = []

        # run the model for a fixed number of glimpses
        for j in range(num_glimpses - 1):
            h_t, cell_state, l_t, log_pi, baseline = model(
                x, l_t, h_t, cell_state, False)

            locations.append(l_t)
            log_pis.append(log_pi)
            baselines.append(baseline)

        # final round of model gives prediction
        h_t, cell_state, l_t, log_pi, baseline, prediction = model(
            x, l_t, h_t, cell_state, is_pred=True)

        locations.append(l_t)
        log_pis.append(log_pi)
        baselines.append(baseline)

        # reinforcement learning term
        log_pis = torch.stack(log_pis).transpose(1, 0)
        baselines = torch.stack(baselines).transpose(1, 0)

        # get predictions for batch
        prediction = torch.squeeze(prediction, dim=1)
        round_prediction = torch.round(prediction)

        # model reward based on correct classification
        reward = (round_prediction.detach() == y).float()

        # want the reward to be repeated for several timesteps
        reward = reward.unsqueeze(1).repeat(1, num_glimpses)

        # compute the action loss
        action_loss = F.binary_cross_entropy(prediction, y.float())
        baseline_loss = F.mse_loss(baselines, reward)

        # adjust reward using baseline (reduce variance)
        adjust_reward = reward - baselines.detach()

        # REINFORCE algo
        reinforce_loss = torch.sum(-log_pis * adjust_reward, dim=1)
        reinforce_loss = torch.mean(reinforce_loss, dim=0)

        # dynamic loss
        loss = action_loss + baseline_loss + reinforce_loss * 0.01

        # get the accuracy for this batch
        acc = torch.sum((round_prediction.detach() ==
                         y).float()) / reward.shape[0] * 100

        # backprop
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        if i % 5 == 0:
            loss_val = loss_sum / 5
            print("LOSS: {}".format(loss_val))
            writer.add_scalar("Loss/train", loss_val,
                              epoch * len(train_loader) + i)

            loss_sum = 0
        # print("CURRENT LOSS: {}\t CURRENT ACCURACY: {}%".format(loss.item(), acc))

    return model


@ torch.no_grad()
def validate_model(val_loader, model, num_valid, writer, epoch):

    num_glimpses = 3

    std = 0.05
    glimpse_size = 512
    location_hidden_size = 128
    glimpse_feature_size = 128
    location_output_size = 2
    hidden_state_size = 256
    # use gpu
    device = torch.device("cuda")

    correct = 0
    total = 0

    for i, (x, y) in enumerate(val_loader):
        batch_size = x.shape[0]
        # send to GPU
        x, y = x.to(device), y.to(device)

       # sample initial location from uniform distribution
        l_t = torch.FloatTensor(batch_size, 2).uniform_(-1, 1).to(device)

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
        for j in range(num_glimpses - 1):
            h_t, cell_state, l_t, log_pi, _ = model(
                x, l_t, h_t, cell_state, False)

        _, _, _, _, _, prediction = model(
            x, l_t, h_t, cell_state, is_pred=True)

        prediction = torch.squeeze(prediction, dim=1)
        round_prediction = torch.round(prediction)

        correct += torch.sum((round_prediction.detach() == y).float())

    acc = correct / num_valid
    print("VALIDATION ACC: {}".format(acc * 100))
