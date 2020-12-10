import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from glimpse_sensor import GlimpseSensor
from inception_pretrained import get_pretrained_inception
from train_network import train, validate_model
from recurrent_attention import RecurrentAttentionModel
from torch.utils.tensorboard import SummaryWriter


def main():

    writer = SummaryWriter()
    # required preprocessing for inception
    transf = transforms.Compose([
        transforms.RandomRotation((0, 360)),
        transforms.RandomResizedCrop(3196),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder("data/train/downsampled", transform=transf)

    validation_perc = 0.1
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_perc * num_train))

    np.random.seed(13)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    batch_size = 16

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=val_sampler
    )

    num_epochs = 100
    # need to confirm these values
    std = 0.05
    glimpse_size = 224
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
        hidden_state_size=hidden_state_size,
        std=std
    )

    num_train = len(train_loader.sampler.indices)
    num_valid = len(val_loader.sampler.indices)
    # use gpu
    device = torch.device("cuda")

    # model = nn.DataParallel(model, dim=0)
    # send model to gpu
    model.to(device)

    # TODO epoch iteration

    # run for a given number of epochs
    for epoch in range(num_epochs):

        print("EPOCH {}".format(epoch))
        # train model
        model = train(train_loader, model, writer, epoch)

        # validate
        validate_model(val_loader, model, num_valid, writer, epoch)

    print("DONE!!!")


if __name__ == "__main__":
    main()
