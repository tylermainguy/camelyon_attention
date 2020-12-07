import torch


class GlimpseSensor:
    """
    Module for the glimpse sensor for the glimpse network. This is responsible
    for generating "glimpses" of a given region from an image.
    """

    def __init__(self, glimpse_size):
        super().__init__()

        self.glimpse_size = glimpse_size
        self.glimpse_factor = glimpse_factor

    def glimpse(self, image, location):
        """
        Function that returns the glimpse of the sensor at a given location.
        For now, this will just be the single image itself. In future iterations,
        it will support several zoom levels.
        """
        dist = self.glimpse_size // 2

        location = self.convert_location(location, image.shape[2])

        x = location[0].numpy().astype("int")[0]
        y = location[1].numpy().astype("int")[0]

        patch = image[:, :, x - dist: x + dist,
                      y - dist: y + dist]

        return patch

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
