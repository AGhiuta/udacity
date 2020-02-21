import cv2
import numpy as np


def draw_epipolar_lines(img, pts2d, F):
    # convert the points to homogeneous coordinates
    pts2d = np.insert(pts2d, 2, 1, axis=1)

    # compute the epipolar lines
    lines = np.dot(F, pts2d.T).T

    # get the lines corresponding to the left & right boundaries of the image
    height, width = img.shape[:2]
    line_l = np.cross([0, 0, 1], [height, 0, 1])
    line_r = np.cross([0, width, 1], [height, width, 1])

    for line in lines:
        pt1 = np.cross(line, line_l)
        pt1 = tuple((pt1 / pt1[-1])[:2].astype(np.int))
        pt2 = np.cross(line, line_r)
        pt2 = tuple((pt2 / pt2[-1])[:2].astype(np.int))

        cv2.line(img, pt1, pt2, [0, 255, 0], 2)

    return img
