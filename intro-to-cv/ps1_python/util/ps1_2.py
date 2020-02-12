import cv2
import numpy as np


def hough_lines_acc(img, thetas=np.arange(-90.0, 90.0)):
    height, width = img.shape
    # Rho and Theta ranges
    thetas -= min(min(thetas), 0)
    diag_len = int(np.ceil(np.sqrt(width**2 + height**2)))   # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some reusable values
    cos_t = np.cos(np.deg2rad(thetas))
    sin_t = np.sin(np.deg2rad(thetas))
    num_thetas = len(thetas)
    num_rhos = len(rhos)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.uint8)
    y_idxs, x_idxs = np.nonzero(img)

    # Vote in the Hough accumulator
    for y, x in zip(y_idxs, x_idxs):
        # Calculate rho. diag_len is added for a positive index
        rho = x * cos_t + y * sin_t + diag_len

        valid_idxs = np.nonzero(rho < num_rhos)
        valid_rhos, valid_thetas = rho[valid_idxs], thetas[valid_idxs]

        c = np.stack([valid_rhos, valid_thetas], axis=1)
        _, idxs, counts = np.unique(
            c, axis=0, return_index=True, return_counts=True)

        accumulator[valid_rhos[idxs].astype(np.uint64),
                    valid_thetas[idxs].astype(np.uint64)] += counts.astype(np.uint64)

    return accumulator, thetas, rhos


def hough_peaks(H, numpeaks=10, threshold=None, nhoodsize=None):
    if not threshold:
        threshold = 0.5 * H.max()

    if not nhoodsize:
        nhoodsize = H.size // 50

    peaks = np.zeros((numpeaks, 3), dtype=np.uint64)
    tmpH = H.copy()

    def clip(x):
        return max(x, 0)

    for i in range(numpeaks):
        # find maximum peak
        r, y, x = np.unravel_index(np.argmax(tmpH, axis=None), tmpH.shape)
        maxVal = tmpH[r, y, x]

        if maxVal > threshold:
            peaks[i] = (r, y, x)
            tmpH[r, y, x] = 0
            k = nhoodsize // 2
            tmpH[clip(r-k):r+k+1, clip(y-k):y+k+1, clip(x-k):x+k+1] = 0

        else:
            return peaks[:i]

    return peaks


def hough_lines_draw(img, filename, peaks, rhos, thetas):
    for peak in peaks:
        rho = rhos[peak[0]]
        theta = thetas[peak[1]]
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        pt0 = rho * np.array([cos_t, sin_t])
        pt1 = tuple((pt0 + 1000 * np.array([-sin_t, cos_t])).astype(int))
        pt2 = tuple((pt0 - 1000 * np.array([-sin_t, cos_t])).astype(int))

        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    cv2.imwrite(filename, img)

    return img
