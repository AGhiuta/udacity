import numpy as np

from util import ps1_2, ps1_5


def filter_lines(peaks, thetas, rhos, theta_threshold, rho_threshold):
    del_list = set()

    for i in range(len(peaks)):
        delta_rho = np.abs(np.array([abs(rhos[peaks[j, 0]] - rhos[peaks[i, 0]])
                                     for j in range(len(peaks))]))
        delta_theta = np.array([abs(thetas[peaks[j, 1]] - thetas[peaks[i, 1]])
                                for j in range(len(peaks))])

        if not ((delta_theta < theta_threshold) & (delta_rho > 1) &
                (delta_rho < rho_threshold)).any():
            del_list.add(i)

    peaks = np.delete(peaks, list(del_list), 0)

    return peaks


def findCircles(img, radius_range=[5, 10], threshold=150, nhoodsize=10):
    radii = np.arange(radius_range[0], radius_range[1])
    nradii = len(radii)
    Hsize = (nradii, ) + img.shape
    H = np.zeros(Hsize, dtype=np.uint64)
    valid_radii = []
    valid_centers = []

    for i, radius in enumerate(radii):
        H[i] = ps1_5.hough_circles_acc(img, radius)

    peaks = ps1_2.hough_peaks(
        H, numpeaks=20, threshold=threshold, nhoodsize=nhoodsize)

    if peaks.size:
        valid_radii, valid_centers = [
            [int(r + radius_range[0]) for r, _, _ in peaks],
            [(y, x) for _, y, x in peaks]
        ]

    return valid_centers, valid_radii
