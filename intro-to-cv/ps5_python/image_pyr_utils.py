import cv2
import numpy as np


def expand(img):
    return cv2.pyrUp(img)


def reduce(img):
    return cv2.pyrDown(img)


def gauss_pyr(img, num_levels=4):
    pyr = [img]

    for _ in range(num_levels-1):
        pyr.append(reduce(pyr[-1]))

    return pyr


def laplace_pyr(img, num_levels=4):
    gpyr = gauss_pyr(img, num_levels=num_levels)
    lpyr = [gpyr[-1]]

    for i in range(num_levels-1, 0, -1):
        G1 = expand(gpyr[i])
        G2 = gpyr[i-1]
        h2, w2 = G2.shape[:2]
        G1 = G1[:h2, :w2]
        h1, w1 = G1.shape[:2]
        G1 = cv2.copyMakeBorder(G1, 0, h2-h1, 0, w2-w1, cv2.BORDER_REPLICATE)

        lpyr.append(cv2.subtract(G2, G1))

    return lpyr[::-1]
