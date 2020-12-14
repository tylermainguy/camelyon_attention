import torch
import torchvision
import cv2
from matplotlib import pyplot as plt


def visualize_glimpse(image, orig, location):
    """
    Function used to visualize a glimpse for the first image in a 
    batch
    """

    # look at first 3 glimpses
    first_set = image[0]
    second_set = image[1]
    third_set = image[2]

    # get first image from the batch
    im1 = first_set[0]
    im2 = second_set[0]
    im3 = third_set[0]

    # permute so channels are last dim
    im1 = im1.permute(1, 2, 0)
    im2 = im2.permute(1, 2, 0)
    im3 = im3.permute(1, 2, 0)

    # original image
    orig = orig[0]
    orig = orig.permute(1, 2, 0)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)

    # original image with location
    ax0.imshow(orig.cpu().numpy())
    ax0.plot(location[0, 1].int().cpu(), location[0, 0].int().cpu(), "go")

    # show glimpses
    ax1.imshow(im1.cpu().numpy())
    ax2.imshow(im2.cpu().numpy())
    ax3.imshow(im3.cpu().numpy())

    plt.show()
