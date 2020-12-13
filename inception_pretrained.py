import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torchvision import datasets, models, transforms


def get_pretrained_inception(out_size):
    """
    Function to get a pretrained inception network ready for use in our network.
    """
    # load in the pretrained model
    model = models.mobilenet_v2(pretrained=True)

    # only want to train the last few layers
    for param in model.parameters():
        param.requires_grad = False

    # get number of features going into last fc layer
    num_features_in = model.classifier[1].in_features

    # replace with our custom fc layer
    model.classifier[1] = nn.Linear(num_features_in, out_size)

    return model
