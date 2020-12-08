import torch
import torchvision
import cv2
from matplotlib import pyplot as plt


def visualize_glimpse(image, location, glimpse):
    print("GLIMPSE SIZE: {}".format(glimpse.shape))
    image = image.permute(1, 2, 0)
    glimpse = glimpse.permute(1, 2, 0)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

    ax0.imshow(image.cpu().numpy())

    ax1.imshow(image.cpu().numpy())
    ax1.plot(location[0].int().cpu(), location[1].int().cpu(), "ro")

    ax2.imshow(glimpse.cpu().numpy())

    plt.show()
