import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
from recurrent_attention import RecurrentAttentionModel


class AverageMeter:
    """
    Tracking of average values used for loss and accuracy monitoring.
    Code taken from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

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


def compute_metrics(log_pis, baselines, prediction, label, params):
    """
    Compute the loss function for the model.
    """

    # get predictions for batch
    prediction = torch.squeeze(prediction, dim=1)

    # round sigmoid val to 1 or 0
    round_prediction = torch.round(prediction)

    # model reward based on correct classification
    reward = (round_prediction.detach() == label).float()

    # want the reward to be repeated for several timesteps
    reward = reward.unsqueeze(1).repeat(1, params["num_glimpses"])

    # compute the action loss
    action_loss = F.binary_cross_entropy(prediction, label.float())

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
                     label).float()) / reward.shape[0] * 100

    return loss, acc


def run_batch(model, x, y, params):
    # get batch size
    batch_size = x.shape[0]

    # visualize what the training data looks like
    # visualize_batch(x, y, batch_size)

    # sample initial location from uniform distribution
    l_t = torch.FloatTensor(
        batch_size, 2).uniform_(-1, 1).to(params["device"])

    # send data to GPU
    x, y = x.to(params["device"]), y.to(params["device"])

    # keep track of locations selected
    locations = []
    log_pis = []
    baselines = []

    # run the model for a fixed number of glimpses
    for j in range(params["num_glimpses"] - 1):
        l_t, log_pi, baseline = model(x, l_t, False)

        locations.append(l_t)
        log_pis.append(log_pi)
        baselines.append(baseline)

    # final round of model gives prediction
    l_t, log_pi, baseline, prediction = model(
        x, l_t, is_pred=True)

    locations.append(l_t)
    log_pis.append(log_pi)
    baselines.append(baseline)

    # stack lists into tensor
    log_pis = torch.stack(log_pis).transpose(1, 0)
    baselines = torch.stack(baselines).transpose(1, 0)

    return locations, log_pis, baselines, prediction


def train(train_loader, model, writer, epoch, params):
    """
    Train the neural network on batches of data.
    """

    # put model in train mode
    model.train()

    # for monitoring loss and accuracy
    losses = AverageMeter()
    accuracy = AverageMeter()

    # use adam optimizer for network
    optimizer = optim.Adam(model.parameters())

    for i, (x, y) in enumerate(train_loader):
        batch_size = x.shape[0]

        optimizer.zero_grad()
        locations, log_pis, baselines, prediction = run_batch(
            model, x, y, params)

        loss, acc = compute_metrics(log_pis, baselines, prediction, y, params)

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
def validate_model(val_loader, model, num_valid, writer, epoch, params):

    # put model in evaluation mode
    model.eval()

    losses = AverageMeter()
    accuracy = AverageMeter()

    for i, (x, y) in enumerate(val_loader):
        batch_size = x.shape[0]

        locations, log_pis, baselines, prediction = run_batch(
            model, x, y, params)

        loss, acc = compute_metrics(log_pis, baselines, prediction, y, params)

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
