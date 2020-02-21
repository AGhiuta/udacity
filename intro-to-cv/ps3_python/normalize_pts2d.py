import numpy as np

from read_file import read_file


def normalize_pts2d(pts2d):
    # compute the mean
    mean = np.mean(pts2d, axis=0)

    # compute the scale
    scale = 1 / np.std(pts2d - mean)

    # compute T as the product of scale and offset matrices
    S = np.diag([scale, scale, 1])
    O = np.array([[1, 0, -mean[0]], [0, 1, -mean[1]], [0, 0, 1]])
    T = np.dot(S, O)

    # convert the points to homogeneous coordinates
    pts2d = np.insert(pts2d, 2, 1, axis=1)

    # normalize the points
    pts2d_norm = np.dot(T, pts2d.T).T

    return T, pts2d_norm[:, :2]
