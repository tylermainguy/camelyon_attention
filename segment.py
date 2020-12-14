import cv2
import math
import numpy as np
import os
import pandas as pd
import time

from matplotlib import pyplot as plt
from openslide import open_slide
from pathlib import Path
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu

# prevent decompression bomb error
Image.MAX_IMAGE_PIXELS = None


def get_bounding_box(slide, downsample_factor):
    """
    Get the bounding box for a sample. Used to reduce the size of sample used,
    as histology samples are typically in the range of gigabytes.
    """

    # get downsampled image for thresholding
    thumbnail = slide.get_thumbnail(
        (slide.dimensions[0] / 256, slide.dimensions[1] / 256))
    thum = np.array(thumbnail)

    # otsu threshold HSV channels
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

    # get coordinates scaled for the downsampled image
    return list(map(lambda x: x // downsample_factor, list(bboxt)))


def downsample(slide, downsample_factor):
    """
    Downsample an image by some given factor
    """
    # gets thumbnail of original size / downsample factor (image size downsampled
    # by some given factor)
    return slide.get_thumbnail(
        (slide.dimensions[0] // downsample_factor, slide.dimensions[1] // downsample_factor))


def main():
    """
    Segmenting WSI data. Takes samples, finds bounding box around the tissue
    within the sample and crops. Image is then zero-padded to the largest
    cropped image size, and scaled to a constant size. 
    """

    # downsample factor for original image
    DOWNSAMPLE_FACTOR = 8

    # hardcoded - largest image size after cropping for 4x downsampling
    x_size = 52096 // DOWNSAMPLE_FACTOR
    y_size = 51136 // DOWNSAMPLE_FACTOR
    # IM_SIZE = (52096, 51136) // DOWNSAMPLE_FACTOR

    # paths
    data_path = Path("data/train/original/tumor/")
    save_path = Path("data/train/downsampled_2/")

    max_x = 0
    max_y = 0

    # for each input image
    for file_path in data_path.rglob("*.tif"):

        start = time.time()

        # get filename, create savepath
        filename = "/".join(str(file_path).split("/")[-2:])
        final_save = save_path / filename
        final_save_path = "/".join(str(final_save).split("/")[:-1])

        # open the given slide
        slide = open_slide(str(file_path))
        print("\tDimensions: {}".format(slide.dimensions))

        # get bounding box around the tissue sample
        coords = get_bounding_box(slide, DOWNSAMPLE_FACTOR)

        # # image size
        x_size = coords[2] - coords[0]
        y_size = coords[3] - coords[1]

        if x_size > max_x:
            max_x = x_size

        if y_size > max_y:
            max_y = y_size

        # get downsampled version of slide
        downsampled = downsample(slide, DOWNSAMPLE_FACTOR)
        print("OUCH")
        # crop downsampled slide around bounding box
        cropped = downsampled.crop(coords)
        downsampled.close()

        # pad image to consistent size
        padded = ImageOps.pad(cropped, (x_size, y_size))
        cropped.close()

        # how long to process a sample
        end = time.time()
        print("\tTime to process: {}".format(end - start))

        resized = padded.resize((4000, 4000))
        padded.close
        # # save the padded image
        resized.save(final_save)
        resized.close()

    print("Max size: ({}, {})".format(max_x, max_y))


if __name__ == "__main__":
    main()
