import torch
import torch.nn.functional as F

from visualize_glimpse import visualize_glimpse


class GlimpseSensor:
    """
    Module for the glimpse sensor for the glimpse network. This is responsible
    for generating "glimpses" of a given region from an image.
    """

    def __init__(self, glimpse_size, num_zooms, zoom_amt):
        super().__init__()

        self.glimpse_size = glimpse_size
        self.num_zooms = num_zooms
        self.zoom_amt = zoom_amt

    def glimpse(self, images, location):
        """
        Function that returns the glimpse of the sensor at a given location.
        For now, this will just be the single image itself. In future iterations,
        it will support several zoom levels.
        """
        dist = self.glimpse_size // 2

        # get total number of images in batch
        batches = images.shape[0]

        n_locs = location.shape[0]
        # get the image size to be used to convert the coordinates
        location = self.convert_location(location, images.shape[2])

        patches = []

        factor = 1

        # get zoom levels for each patch in the batch
        #   poet, and I didn't even know it

        for i in range(self.num_zooms):
            patch = self.get_zoomed_patch(images, location, factor)
            patches.append(patch)
            factor = factor * self.zoom_amt

        for i in range(len(patches)):
            zoom_factor = patches[i].shape[-1] // self.glimpse_size
            patches[i] = F.avg_pool2d(patches[i], zoom_factor)

        # visualize_glimpse(patches, location)

        patches = torch.cat(patches, 1)
        patches = patches.view(patches.shape[0], -1)
        # if (i == 0):

        return patches

    def get_zoomed_patch(self, img, location, factor):
        """
        Gets the patch for a given image at a location l, with scaling size.
        """

        total_size = self.glimpse_size * factor

        dist = total_size // 2

        img = F.pad(img, (dist, dist, dist, dist))

        patches = []
        # over each image in the batch
        for i in range(location.shape[0]):
            # get coordinates of glimpse
            x = location[i, 0].int()
            y = location[i, 1].int()

            # middle location + padding
            x_start = x + dist
            y_start = y + dist

            # extract the patch
            patch = img[i, :, x_start - dist: x_start + dist,
                        y_start - dist: y_start + dist]

            # add the patch to the current set
            patches.append(patch)

        # return list of patches as a tensor
        return torch.stack(patches)

    def convert_location(self, location, image_size):
        """
        Image coordinates need to be represented as a value in the range of
        [-1, 1]. In order to index the given image, these coordinates need
        to be scaled to [0, img_size] where img_size is the size of the image
        """
        # gives range [0, 2*image_size]
        loc_2 = (1 + location) * image_size

        # return values in range [0, image_size]
        return (loc_2 * 0.5).long()
