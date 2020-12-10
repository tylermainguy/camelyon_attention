import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
from recurrent_attention import RecurrentAttentionModel

# https://github.com/pytorch/examples/blob/master/imagenet/main.py


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, writer, epoch):
    """
    Train the neural network on batches of data.
    """
    losses = AverageMeter()
    accuracy = AverageMeter()

    num_glimpses = 5

    # need to confirm these values
    hidden_state_size = 256

    # use gpu
    device = torch.device("cuda")

    # use adam optimizer for network
    optimizer = optim.Adam(model.parameters())

    for i, (x, y) in enumerate(train_loader):
        print("BATCH {}".format(i))
        # get batch size
        batch_size = x.shape[0]

        # visualize what the training data looks like
        # visualize_batch(x, y, batch_size)

        # sample initial location from uniform distribution
        l_t = torch.FloatTensor(batch_size, 2).uniform_(-1, 1).to(device)

        # intialize hidden state for LSTM
        h_t = torch.randn(
            batch_size,
            1,
            hidden_state_size,
            dtype=torch.float,
            device=device,
            requires_grad=True,
        )

        # intialize cell state for LSTM
        cell_state = torch.randn(
            batch_size,
            1,
            hidden_state_size,
            dtype=torch.float,
            device=device,
            requires_grad=True
        )

        # zero model gradients before starting pass
        optimizer.zero_grad()

        # send data to GPU
        x, y = x.to(device), y.to(device)

        # keep track of locations selected
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

        # stack list into tensor
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

        # compute baseline loss for variance reduction
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

        # update tracking of loss
        losses.update(loss, batch_size)
        accuracy.update(acc, batch_size)

        # backprop
        loss.backward()
        optimizer.step()

        iteration = epoch * len(train_loader) + i

        # tensorboard logging
        writer.add_scalar("Loss/train", losses.avg, iteration)
        writer.add_scalar("Accuracy/train", accuracy.avg, iteration)

    return model


@ torch.no_grad()
def validate_model(val_loader, model, num_valid, writer, epoch):

    losses = AverageMeter()
    accuracy = AverageMeter()

    num_glimpses = 5

    location_hidden_size = 128
    glimpse_feature_size = 128
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
            batch_size,
            1,
            hidden_state_size,
            dtype=torch.float,
            device=device,
            requires_grad=True,
        )

        cell_state = torch.randn(
            batch_size,
            1,
            hidden_state_size,
            dtype=torch.float,
            device=device,
            requires_grad=True
        )

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

        losses.update(loss, batch_size)
        accuracy.update(acc, batch_size)

        iteration = epoch * len(val_loader) + i

        writer.add_scalar("Loss/validation", losses.avg, iteration)
        writer.add_scalar("Accuracy/validation", accuracy.avg, iteration)


def visualize_batch(data, labels, batch_size):

    fig = plt.figure()

    for i in range(1, 4 * 4 + 1):
        img = data[i - 1, :, :, :]
        img = img.permute(1, 2, 0)
        fig.add_subplot(4, 4, i)
        plt.imshow(img)

    plt.show()
