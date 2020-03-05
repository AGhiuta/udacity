import cv2
import numpy as np


def compute_x_grad(img, ksize=None, normalize=False):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)

    if normalize:
        grad_x = cv2.normalize(grad_x, grad_x, 0, 255,
                               cv2.NORM_MINMAX, cv2.CV_8U)

    return grad_x


def compute_y_grad(img, ksize=None, normalize=False):
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    if normalize:
        grad_y = cv2.normalize(grad_y, grad_y, 0, 255,
                               cv2.NORM_MINMAX, cv2.CV_8U)

    return grad_y


def compute_angle(grad_x, grad_y):
    return np.arctan2(grad_y, grad_x)
