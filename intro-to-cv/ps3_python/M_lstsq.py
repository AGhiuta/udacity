import numpy as np


def M_lstsq(pts2d, pts3d):
    num_pts = pts2d.shape[0]
    A = np.empty((2*num_pts, 11))
    b = np.empty((2*num_pts, 1))
    zeros = np.zeros((num_pts, 1))
    ones = np.ones((num_pts, 1))

    x, y = np.split(pts2d, 2, axis=1)
    X, Y, Z = np.split(pts3d, 3, axis=1)

    A[::2, :] = np.hstack(
        (X, Y, Z, ones, zeros, zeros, zeros, zeros, -x*X, -x*Y, -x*Z))
    A[1::2, :] = np.hstack(
        (zeros, zeros, zeros, zeros, X, Y, Z, ones, -y*X, -y*Y, -y*Z))
    b[::2, :] = x
    b[1::2, :] = y

    M, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return np.reshape(np.append(M, 1), (3, 4))
