import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from average_meter import AverageMeter

from matplotlib import pyplot as plt
from recurrent_attention import RecurrentAttentionModel


def compute_metrics(log_pis, baselines, probabs, label, params):
    """
    Compute the loss function for the model.
    """

    # # get predictions for batch
    # prediction = torch.squeeze(prediction, dim=1)

    # round sigmoid val to 1 or 0
    predicted = torch.max(probabs, 1)[1]

    # model reward based on correct classification
    reward = (predicted.detach() == label).float()

    # want the reward to be repeated for several timesteps
    reward = reward.unsqueeze(1).repeat(1, params["num_glimpses"])

    # compute the action loss
    action_loss = F.nll_loss(probabs, label)

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
    acc = torch.sum((predicted.detach() ==
                     label).float()) / reward.shape[0] * 100

    return loss, acc


def run_batch(model, x, y, params):
    # get batch size
    batch_size = x.shape[0]

    h_t = torch.zeros(
        params["batch_size"],
        params["hidden_state_size"],
        dtype=torch.float,
        device=params["device"],
        requires_grad=True,
    )
    # visualize what the training data looks like
    # visualize_batch(x, y, batch_size)

    # sample initial location from uniform distribution
    l_t = torch.FloatTensor(
        batch_size, 2).uniform_(-1, 1).to(params["device"])
    l_t.requires_grad = True
    # keep track of locations selected
    locations = []
    log_pis = []
    baselines = []

    # run the model for a fixed number of glimpses
    for j in range(params["num_glimpses"] - 1):
        l_t, h_t, log_pi, baseline = model(x, l_t, h_t, False)

        locations.append(l_t)
        log_pis.append(log_pi)
        baselines.append(baseline)

    # final round of model gives prediction
    l_t, h_t, log_pi, baseline, prediction = model(x, l_t, h_t, is_pred=True)

    locations.append(l_t)
    log_pis.append(log_pi)
    baselines.append(baseline)

    # stack lists into tensor
    log_pis = torch.stack(log_pis).transpose(1, 0)
    baselines = torch.stack(baselines).transpose(1, 0)

    return locations, log_pis, baselines, prediction


def train(train_loader, model, writer, epoch, params, optimizer):
    """
    Train the neural network on batches of data.
    """

    # put model in train mode
    model.train()

    # for monitoring loss and accuracy
    losses = AverageMeter()
    accuracy = AverageMeter()

    for i, (x, y) in enumerate(train_loader):  # send data to GPU
        print(torch.isnan(x))
        x, y = x.to(params["device"]), y.to(params["device"])
        batch_size = x.shape[0]

        optimizer.zero_grad()
        locations, log_pis, baselines, prediction = run_batch(
            model, x, y, params)

        # compute loss and accuracy
        loss, acc = compute_metrics(log_pis, baselines, prediction, y, params)

        # update tracking of loss
        losses.update(loss.item(), batch_size)
        accuracy.update(acc.item(), batch_size)

        print("LOSS: ", loss.item())
        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        for p in model.parameters():
            print("GRAD: {}".format(p.grad.norm()))
        # plot_grad_flow(model.named_parameters())
        optimizer.step()

        # tensorboard logging
        iteration = epoch * len(train_loader) + i
        writer.add_scalar("Loss/train", losses.avg, iteration)
        writer.add_scalar("Accuracy/train", accuracy.avg, iteration)


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], color="c", lw=4),
                plt.Line2D([0], [0], color="b", lw=4),
                plt.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.show()


@ torch.no_grad()
def validate_model(val_loader, model, num_valid, writer, epoch, params):

    # put model in evaluation mode
    model.eval()

    losses = AverageMeter()
    accuracy = AverageMeter()

    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(params["device"]), y.to(params["device"])
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
