import cv2
import numpy as np


def canny(img, sigma=0.5, low=None, high=None):
    mean = img.mean()

    if not low:
        low = max(0, (1.0 - sigma) * mean)

    if not high:
        high = min(255, (1.0 + sigma) * mean)

    img_edges = cv2.Canny(img, low, high)

    return img_edges
