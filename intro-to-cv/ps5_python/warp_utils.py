import cv2
import numpy as np


def warp_flow(frame, op_flow):
    h, w = op_flow.shape[:2]
    flow_map = -op_flow.copy()
    flow_map[:, :, 0] += np.arange(w)
    flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]

    return cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)
