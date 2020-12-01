import cv2
import math
import numpy
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from openslide import open_slide
from PIL import Image
import time


def get_bounding_box(slide):
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
    bboxt = math.floor(np.min(xxmin)*256), math.floor(np.max(xxmax) *
                                                      256), math.floor(np.min(yymin)*256), math.floor(np.max(yymax)*256)

    return bboxt


def main():
    """
    Main function for segmenting images.
    """

    slide = open_slide("data/normal_001.tif")


if __name__ == "__main__":
    main()
