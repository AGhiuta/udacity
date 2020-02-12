import cv2
import numpy as np


def hough_circles_acc(img, radius):
    thetas = np.deg2rad(np.arange(0, 360))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    height, width = img.shape

    # Hough accumulator array
    accumulator = np.zeros(img.shape, dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)

    # Vote in the Hough accumulator
    for y, x in zip(y_idxs, x_idxs):
        # Compute a and b
        a = (y - radius * sin_t + .5).astype(np.uint64)
        b = (x - radius * cos_t + .5).astype(np.uint64)
        valid_idxs = np.nonzero((a < height) & (b < width))
        a, b = a[valid_idxs], b[valid_idxs]

        c = np.stack([a, b], axis=1)
        _, idxs, counts = np.unique(
            c, axis=0, return_index=True, return_counts=True)
        accumulator[a[idxs], b[idxs]] += counts.astype(np.uint64)

    return accumulator


def hough_circles_draw(img, filename, centers, radius):
    for center in centers:
        cv2.circle(img, center[::-1], radius, (0, 255, 0), 2)

    cv2.imwrite(filename, img)

    return img
