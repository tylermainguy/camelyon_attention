import torch
import torch.nn.functional as F

from visualize_glimpse import visualize_glimpse


class GlimpseSensor:
    """
    Module for the glimpse sensor for the glimpse network. This is responsible
    for generating "glimpses" of a given region from an image.
    """

    def __init__(self, glimpse_size):
        super().__init__()

        self.glimpse_size = glimpse_size

    def glimpse(self, images, location):
        """
        Function that returns the glimpse of the sensor at a given location.
        For now, this will just be the single image itself. In future iterations,
        it will support several zoom levels.
        """
        dist = self.glimpse_size // 2

        # get total number of images in batch
        batches = images.shape[0]

        # get the image size to be used to convert the coordinates
        location = self.convert_location(location, images.shape[2])

        patches = []
        for i in range(batches):
            x = location[i, 0].int()
            y = location[i, 1].int()

            current_img = images[i, :, :, :]

            padded_img = F.pad(current_img, (dist, dist, dist, dist))
            x_start = x + dist
            y_start = y + dist

            # print("LOCATION: ({}, {})".format(x_start, y_start))
            patch = padded_img[:, x_start - dist: x_start + dist,
                               y_start - dist: y_start + dist]

            if (i == 0):
                visualize_glimpse(padded_img, (x_start, y_start), patch)
                patches.append(patch)

        patches = torch.stack(patches, dim=0)
        return patches

    def convert_location(self, location, image_size):
        """
        Image coordinates need to be represented as a value in the range of
        [-1, 1]. In order to index the given image, these coordinates need
        to be scaled to [0, img_size] where img_size is the size of the image
        """
        # gives range [0, 2*image_size]
        loc_2 = (1 + location) * image_size

        # return values in range [0, image_size]
        return loc_2 * 0.5
