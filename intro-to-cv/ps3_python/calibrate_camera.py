import numpy as np
import random

from M_lstsq import M_lstsq


def calibrate_camera(pts2d, pts3d,
                     num_calibration_pts=8,
                     num_test_pts=4,
                     num_iter=10):
    num_pts = pts2d.shape[0]
    res_min = float('inf')
    M_best = None

    for i in range(num_iter):
        idxs_calib = random.sample(range(num_pts), num_calibration_pts)
        idxs_test = random.sample(
            [i for i in range(num_pts) if i not in idxs_calib], num_test_pts)

        # get the points used for calibration
        pts2d_calib = pts2d[idxs_calib]
        pts3d_calib = pts3d[idxs_calib]

        # get the points used for testing
        pts2d_test = pts2d[idxs_test]
        pts3d_test = pts3d[idxs_test]

        M = M_lstsq(pts2d_calib, pts3d_calib)

        # compute the projections using M and calculate the residual
        pts2d_proj = np.dot(M, np.insert(pts3d_test, 3, 1, axis=1).T).T
        pts2d_proj /= pts2d_proj[:, 2:]
        res = np.linalg.norm(pts2d_test[:, :2] - pts2d_proj[:, :2])

        if res < res_min:
            res_min = res
            M_best = M.copy()

    return res_min, M_best
