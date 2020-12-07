import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, models, transforms
import torchvision


def get_pretrained_inception(out_size):
    """
    Function to get a pretrained inception network ready for use in our network.
    """
    # load in the pretrained model
    model = models.resnet18(pretrained=True)

    # only want to train the last few layers
    for param in model.parameters():
        param.requires_grad = False

    # get number of features going into last fc layer
    num_features_in = model.fc.in_features

    # replace with our custom fc layer
    model.fc = nn.Linear(num_features_in, out_size)

    return model
