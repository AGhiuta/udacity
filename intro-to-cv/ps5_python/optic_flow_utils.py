import cv2
import numpy as np

from image_pyr_utils import *
from warp_utils import *


def lk_optic_flow(frame1, frame2, wsize=2):
    """
    Implementation of Lucas-Kanade optic flow algorithm
    taken from: https://stackoverflow.com/questions/14321092/lucas-kanade-python-numpy-implementation-uses-enormous-amount-of-memory
    """
    assert frame1.shape == frame2.shape

    [Ix, Iy, It] = [np.zeros(frame1.shape, dtype=np.float) for _ in range(3)]
    Ix[1:-1, 1:-1] = cv2.subtract(frame1[1:-1, 2:], frame1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = cv2.subtract(frame1[2:, 1:-1], frame1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = cv2.subtract(frame1[1:-1, 1:-1], frame2[1:-1, 1:-1])

    # w = np.zeros((wsize, wsize), dtype=np.float)
    # w[wsize//2, wsize//2] = 1
    # w = cv2.GaussianBlur(w, (wsize, wsize), 1)

    # Ix = cv2.filter2D(Ix, -1, w)
    # Iy = cv2.filter2D(Iy, -1, w)
    # It = cv2.filter2D(It, -1, w)

    params = np.zeros(frame1.shape + (5,), dtype=np.float)
    params[..., 0] = Ix**2
    params[..., 1] = Iy**2
    params[..., 2] = Ix * Iy
    params[..., 3] = Ix * It
    params[..., 4] = Iy * It

    params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    params = (params[2*wsize+1:, 2*wsize+1:] -
              params[2*wsize+1:, :-1-2*wsize] -
              params[:-1-2*wsize, 2*wsize+1:] +
              params[:-1-2*wsize, :-1-2*wsize])

    op_flow = np.zeros(frame1.shape + (2,))
    det = params[..., 0] * params[..., 1] - params[..., 2]**2

    op_flow[wsize+1:-1-wsize, wsize+1:-1-wsize, 0] = \
        np.where(det != 0,
                 (params[..., 1] * params[..., 3] -
                  params[..., 2] * params[..., 4]) / det, 0)[:-1, :-1]
    op_flow[wsize+1:-1-wsize, wsize+1:-1-wsize, 1] = \
        np.where(det != 0,
                 (params[..., 0] * params[..., 4] -
                  params[..., 2] * params[..., 3]) / det, 0)[:-1, :-1]

    return op_flow.astype(np.float32)


def hlk_optic_flow(imgs, num_levels=4, wsize=15):
    pyrL, pyrR = [gauss_pyr(img, num_levels=num_levels) for img in imgs]

    for k in range(num_levels-1, -1, -1):
        if k == num_levels-1:
            op_flow = np.zeros(pyrL[k].shape + (2,), dtype=np.float32)
        else:
            h1, w1 = pyrR[k].shape
            op_flow = 2 * expand(op_flow)
            op_flow = op_flow[:h1, :w1, :]
            h2, w2 = op_flow.shape[:2]
            op_flow = cv2.copyMakeBorder(
                op_flow, 0, h1-h2, 0, w1-w2, cv2.BORDER_REPLICATE)

        wL = warp_flow(pyrL[k], op_flow)
        op_flow += lk_optic_flow(wL, pyrR[k], wsize=wsize)

    return op_flow
