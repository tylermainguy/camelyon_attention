import numpy as np
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn

from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from inception_pretrained import get_pretrained_inception
from recurrent_attention import RecurrentAttentionModel
from train_network import train, validate_model


def get_params():
    """
    Set the parameters for various aspects of the program.
    """

    params = {}

    params["multi_gpu"] = False
    params["load_model"] = False
    params["test"] = False
    params["visualize_batch"] = False
    params["batch_size"] = 16
    params["num_epochs"] = 500
    params["glimpse_size"] = 64
    params["std"] = 0.1

    params["validation_perc"] = 0.1
    # size["parameters
    params["location_hidden_size"] = 128
    params["glimpse_feature_size"] = 256
    params["glimpse_hidden"] = 128
    params["num_patches"] = 3
    params["zoom_amt"] = 2
    params["location_output_size"] = 2
    params["hidden_state_size"] = 256
    params["channels"] = 3
    params["num_glimpses"] = 5
    params["device"] = torch.device("cuda:2")

    return params


def get_dataset(params):
    """
    Prepare dataset for use by the model. Performs several different data
    augmentation tasks; random rotation, random flipping, and random cropping.
    The images are then split into train and validation sets.
    """

    # data augmentation and preprocessing
    transf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    # load dataset
    dataset = datasets.ImageFolder("data/train/downsampled", transform=transf)

    # 10% validation set
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(params["validation_perc"] * num_train))

    # randomize selection of train/val
    np.random.seed(1)
    np.random.shuffle(indices)

    # get train/val indices
    train_idx, valid_idx = indices[split:], indices[: split]

    # train/val split
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    # prepare train dataset loader
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params["batch_size"],
        sampler=train_sampler,
        drop_last=True
    )

    # prepare validation set loader
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params["batch_size"],
        sampler=val_sampler,
        drop_last=True
    )

    return train_loader, val_loader


def get_model(params):
    """
    Get the model. Can either instantiate an instance of the model based
    on parameters given, or load a previously trained model from memory.
    """

    # load previous model
    if params["load_model"] and os.path.exists("checkpoints/2020-12-11.pth"):
        model = torch.load("checkpoints/2020-12-11.pth")

    # create model
    else:
        model = RecurrentAttentionModel(
            glimpse_size=params["glimpse_size"],
            location_hidden_size=params["location_hidden_size"],

            glimpse_feature_size=params["glimpse_feature_size"],
            location_output_size=params["location_output_size"],
            hidden_state_size=params["hidden_state_size"],
            std=params["std"],
            glimpse_hidden=params["glimpse_hidden"],
            channels=params["channels"],
            num_patches=params["num_patches"],
            zoom_amt=params["zoom_amt"]
        )

    return model


def main():
    """
    Main program for training and validating the recurrent attention model (RAM)
    applied to the problem of WSI classification.
    """
    # get params for this run
    params = get_params()

    # for tensorboard logging
    writer = SummaryWriter()

    # get train and validation datasets
    train_loader, val_loader = get_dataset(params)

    # get model
    model = get_model(params)

    if params["multi_gpu"]:
        model = nn.DataParallel(model, dim=0)
    model.to(params["device"])

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # iterate over epochs
    for epoch in range(params["num_epochs"]):

        start = time.time()
        print("EPOCH {}".format(epoch))
        train(train_loader, model, writer, epoch, params, optimizer)

        end = time.time()
        print("\t...completed in {} seconds".format(end - start))
        # validate
        validate_model(val_loader, model, writer, epoch, params)

    # save model after training completes
    torch.save(model, "checkpoints/2020-12-14.pth")

    # TODO if we're running model on test set
    if params["test"]:
        # test_model(test_loader, model, writer)
        print("testing")


if __name__ == "__main__":
    main()
