import argparse
from typing import List

import cv2
import matplotlib.pyplot as plt
from path import Path

from collections import namedtuple
from typing import List

import cv2
import numpy as np

from .word_detector import prepare_img, detect_words


def findWords(fn_img, kernel_size=25, sigma=11, theta=7, min_area=100, img_height=1000):

    # load image and process it
    img = prepare_img(fn_img, img_height)
    res = detect_words(img,
                        kernel_size=kernel_size,
                        sigma=sigma,
                        theta=theta,
                        min_area=min_area)

    # plot results
    plt.imshow(img, cmap='gray')
    for det in res:
        xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
        ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
        plt.plot(xs, ys)
    plt.show()