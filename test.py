import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt


def test(img):
    """
    Load data/001-rgb.png image then binarize it
    """
    img = cv2.imread('data/' + img + '-rgb.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # plt.imshow(thresh, cmap='gray')
    # plt.show()
    return thresh


def median(img):
    """
    Median filter
    """
    img = cv2.medianBlur(img, 5)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img


def erode(img):
    """
    Erode image
    """
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img


def dilate(img):
    """
    Dilate image
    """
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img


def contours(img):
    """
    Detect vertical lines
    """
    edges = cv2.Canny(img, 100, 200)
    return edges


def max_area(num_labels, labels, stats):
    list = []
    for i in range(num_labels):
        list.append(stats[i][4])
    list.sort()
    return list[-2]


def imshow_components(img):
    # Map component labels to hue val
    output = cv2.connectedComponentsWithStats(img)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    area = max_area(num_labels, labels, stats)
    print(area)
    for label in range(1, num_labels):
        if stats[label][4] == area:
            mask = np.array(labels, dtype=np.uint8)
            mask[labels == label] = 255
            plt.imshow(mask, cmap='gray')
            plt.show()
            return mask


if __name__ == '__main__':
    img = test(sys.argv[1])
    img = erode(img)
    # img = dilate(img)
    img = median(img)
    img = contours(img)
    img = imshow_components(img)
