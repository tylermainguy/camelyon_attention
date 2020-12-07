import cv2
import os
import math
import numpy
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from openslide import open_slide, ImageSlide
from PIL import Image
from pathlib import Path
import time


def get_bounding_box(slide, downsample_factor):
    """
    Get the bounding box for a sample. Used to reduce the size of sample used,
    as histology samples are typically in the range of gigabytes.
    """

    # get thumbnail of image (we can downsample here for thresholding)
    thumbnail = slide.get_thumbnail(
        (slide.dimensions[0] / 256, slide.dimensions[1] / 256))
    thum = np.array(thumbnail)

    # convert image to hsv for otsu thresholding
    hsv_image = cv2.cvtColor(thum, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    hthresh = threshold_otsu(h)
    sthresh = threshold_otsu(s)
    vthresh = threshold_otsu(v)

    # get hsv values for thresholding
    minhsv = np.array([hthresh, sthresh, 70], np.uint8)
    maxhsv = np.array([180, 255, vthresh], np.uint8)
    thresh = [minhsv, maxhsv]

    # extracting contours to get bounding box
    rgbbinary = cv2.inRange(hsv_image, thresh[0], thresh[1])
    contours, _ = cv2.findContours(
        rgbbinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxtcols = ['xmin', 'xmax', 'ymin', 'ymax']
    bboxt = pd.DataFrame(columns=bboxtcols)

    # get locations of each contour to be used for creating bounding box
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        bboxt = bboxt.append(
            pd.Series([x, x+w, y, y+h], index=bboxtcols), ignore_index=True)
        bboxt = pd.DataFrame(bboxt)

    # get bounding box of sample
    xxmin = list(bboxt['xmin'])
    xxmax = list(bboxt['xmax'])
    yymin = list(bboxt['ymin'])
    yymax = list(bboxt['ymax'])
    bboxt = math.floor(np.min(xxmin)*256), math.floor(np.min(yymin) *
                                                      256), math.floor(np.max(xxmax)*256), math.floor(np.max(yymax)*256)

    return list(map(lambda x: x // downsample_factor, list(bboxt)))


def downsample(slide, downsample_factor):
    """
    Downsample an image by some given factor
    """
    # gets thumbnail of original size / downsample factor (image size downsampled
    # by some given factor)
    return slide.get_thumbnail(
        (slide.dimensions[0] / downsample_factor, slide.dimensions[1] / downsample_factor))


def main():
    """
    Main function for segmenting images.
    """

    DOWNSAMPLE_FACTOR = 16

    data_path = Path("data/train/original")
    save_path = Path("data/train/downsampled/")

    max_x = 0
    max_y = 0

    for file_path in data_path.rglob("*.tif"):
        # get filename, create savepath
        filename = "/".join(str(file_path).split("/")[-2:])
        final_save = save_path / filename
        final_save_path = "/".join(str(final_save).split("/")[:-1])

        # open the given slide
        slide = open_slide(str(file_path))

        # get bounding box around the tissue sample
        coords = get_bounding_box(slide, DOWNSAMPLE_FACTOR)

        # get downsampled version of slide
        downsampled = downsample(slide, DOWNSAMPLE_FACTOR)

        # crop downsampled slide around bounding box
        cropped = downsampled.crop(coords)

        if not os.path.exists(final_save_path):
            os.makedirs(final_save_path)

        x, y = cropped.size

        print("Image size: {}".format(cropped.size))
        if x > max_x:
            max_x = x

        if y > max_y:
            max_y = y
        # don't want to just crop. need to get some consistent "largest size",
        # and determine from there what the size of the images should be
        # print("Image shape: {}".format(cropped.size))
        # # maybe combine this with the crop
        # cropped = cropped.resize((2000, 2000))
        cropped.save(final_save)

    print("Max width: {}".format(max_x))
    print("Max height: {}".format(max_y))


if __name__ == "__main__":
    main()
