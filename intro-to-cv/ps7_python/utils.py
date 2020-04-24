import cv2
import numpy as np


def compute_frame_diff_seq(vname, num_frames=10, theta=127,
                           blur_ksize=(3, 3), blur_sigma=1,
                           morph_ksize=(3, 3)):
    cap = cv2.VideoCapture(vname)
    ret, frame1 = cap.read()

    assert ret == True

    frame_diff_seq = []
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.GaussianBlur(frame1, blur_ksize, blur_sigma)
    morph_kernel = np.ones(morph_ksize, dtype=np.uint8)

    while cap.isOpened() and num_frames > 0:
        ret, frame2 = cap.read()
        num_frames -= 1

        if not ret:
            break

        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.GaussianBlur(frame2, blur_ksize, blur_sigma)
        diff = (np.abs(cv2.subtract(frame2, frame1)) >= theta).astype(np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, morph_kernel)
        frame1 = frame2

        frame_diff_seq.append(diff)

    cap.release()

    return frame_diff_seq


def compute_mhi_seq(frame_diff_seq, tau=10, t_end=10):
    M = np.zeros(frame_diff_seq[0].shape, dtype=np.float)
    ones = np.ones(M.shape)

    for t, B in enumerate(frame_diff_seq):
        M = tau * (B == 1) + np.clip(np.subtract(M, ones),
                                     0, 255) * (B == 0)

        if t == t_end:
            break

    return M


def compute_hu_moments(img):
    h, w = img.shape
    pq = [[2, 0], [0, 2], [1, 2], [2, 1], [2, 2], [3, 0], [0, 3]]
    M_00 = np.sum(img)
    M_01 = np.sum(np.arange(h).reshape((-1, 1)) * img)
    M_10 = np.sum(np.arange(w) * img)
    x_mean = M_10 / M_00
    y_mean = M_01 / M_00

    µ, η = [], []

    for (p, q) in pq:
        cx = (np.arange(w) - x_mean)**p
        cy = ((np.arange(h) - y_mean) ** q).reshape((-1, 1))
        µ.append(np.sum(cx * cy * img))
        η.append(µ[-1] / M_00**(1 + 0.5*(p+q)))

    return µ, η
