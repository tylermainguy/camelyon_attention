import torch
import torchvision
import cv2
from matplotlib import pyplot as plt


def visualize_glimpse(image, location):

    first_set = image[0]
    second_set = image[1]
    third_set = image[2]

    im1 = first_set[0]
    im2 = second_set[0]
    im3 = third_set[0]

    im1 = im1.permute(1, 2, 0)
    im2 = im2.permute(1, 2, 0)
    im3 = im3.permute(1, 2, 0)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

    ax0.imshow(im1.cpu().numpy())

    ax1.imshow(im2.cpu().numpy())
    # ax1.plot(location[1].int().cpu(), location[0].int().cpu(), "ro")

    ax2.imshow(im3.cpu().numpy())

    plt.show()
