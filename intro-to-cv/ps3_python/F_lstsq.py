import numpy as np


def F_lstsq(pts2d_a, pts2d_b):
    ua, va = np.split(pts2d_a, 2, axis=1)
    ub, vb = np.split(pts2d_b, 2, axis=1)

    A = np.hstack((ua*ub, va*ub, ub, ua*vb, va*vb, vb, ua, va))
    b = -np.ones((pts2d_a.shape[0], ), dtype=np.float)
    F, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return np.reshape(np.append(F, 1), (3, 3))
