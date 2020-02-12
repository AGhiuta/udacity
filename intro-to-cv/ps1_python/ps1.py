"""
Problem Set 1: Edges and Lines
"""

import cv2
import os
import numpy as np

from pathlib import Path
from util import ps1_1, ps1_2, ps1_5, ps1_utils


def problem2(imdir):
    # Load the input grayscale image as img and generate an edge image
    img = cv2.imread(os.path.join(
        imdir, "input/ps1-input0.png"), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = ps1_1.canny(img_gray)

    cv2.imwrite(os.path.join(imdir, "output/ps1-1-a-1.png"), img_edges)

    # Compute the Hough Transform for lines and produce an accumulator array.
    H, thetas, rhos = ps1_2.hough_lines_acc(img_edges)
    H = cv2.normalize(H, H, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    cv2.imwrite(os.path.join(imdir, "output/ps1-2-a-1.png"), H)

    # Find indices of the accumulator array (here line parameters) that
    # correspond to local maxima
    H_peaks = np.reshape(H, (1, ) + H.shape)
    peaks = ps1_2.hough_peaks(H_peaks, nhoodsize=5)[:, 1:]

    for peak in peaks:
        cv2.circle(H_peaks[0], peak[::-1], 5, (255, 255, 255), -1)

    cv2.imwrite(os.path.join(imdir, "output/ps1-2-b-1.png"), H_peaks[0])

    _ = ps1_2.hough_lines_draw(img, os.path.join(
        imdir, "output/ps1-2-c-1.png"), peaks, rhos, thetas)


def problem3(imdir):
    img_noisy = cv2.imread(os.path.join(
        imdir, "input/ps1-input0-noise.png"), cv2.IMREAD_COLOR)
    img_noisy_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)
    img_smooth_gray = cv2.GaussianBlur(img_noisy_gray, (17, 17), 3)

    cv2.imwrite(os.path.join(imdir, "output/ps1-3-a-1.png"), img_smooth_gray)

    img_noisy_edges = ps1_1.canny(img_noisy_gray, low=23, high=70)
    img_smooth_edges = ps1_1.canny(img_smooth_gray, low=23, high=70)

    cv2.imwrite(os.path.join(imdir, "output/ps1-3-b-1.png"), img_noisy_edges)
    cv2.imwrite(os.path.join(imdir, "output/ps1-3-b-2.png"), img_smooth_edges)

    H_smooth, thetas_smooth, rhos_smooth = ps1_2.hough_lines_acc(
        img_smooth_edges)
    H_smooth = cv2.normalize(H_smooth, H_smooth, 0, 255,
                             cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    H_peaks_smooth = np.reshape(H_smooth, (1, ) + H_smooth.shape)
    peaks_smooth = ps1_2.hough_peaks(H_peaks_smooth, nhoodsize=10)[:, 1:]

    for peak in peaks_smooth:
        cv2.circle(H_peaks_smooth[0], peak[::-1], 5, (255, 255, 255), -1)

    cv2.imwrite(os.path.join(imdir, "output/ps1-3-c-1.png"), H_peaks_smooth[0])

    _ = ps1_2.hough_lines_draw(img_noisy, os.path.join(
        imdir, "output/ps1-3-c-2.png"), peaks_smooth, rhos_smooth, thetas_smooth)


def problem4(imdir):
    img = cv2.imread(os.path.join(
        imdir, "input/ps1-input1.png"), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_smooth = cv2.GaussianBlur(img_gray, (11, 11), 3)
    cv2.imwrite(os.path.join(imdir, "output/ps1-4-a-1.png"), img_smooth)

    img_edges = ps1_1.canny(img_smooth)
    cv2.imwrite(os.path.join(imdir, "output/ps1-4-b-1.png"), img_edges)

    H, thetas, rhos = ps1_2.hough_lines_acc(img_edges)
    H = cv2.normalize(H, H, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    H_peaks = np.reshape(H, (1, ) + H.shape)
    peaks = ps1_2.hough_peaks(H_peaks, nhoodsize=15)[:, 1:]

    for peak in peaks:
        cv2.circle(H_peaks[0], peak[::-1], 5, (255, 255, 255), -1)

    cv2.imwrite(os.path.join(imdir, "output/ps1-4-c-1.png"), H_peaks[0])

    _ = ps1_2.hough_lines_draw(np.dstack((img_gray, img_gray, img_gray)),
                               os.path.join(imdir, "output/ps1-4-c-2.png"),
                               peaks, rhos, thetas)


def problem5(imdir):
    img = cv2.imread(os.path.join(
        imdir, "input/ps1-input1.png"), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_smooth = cv2.GaussianBlur(img_gray, (11, 11), 3)
    img_circles = np.dstack((img_gray, img_gray, img_gray))
    cv2.imwrite(os.path.join(imdir, "output/ps1-5-a-1.png"), img_smooth)

    img_edges = ps1_1.canny(img_smooth)
    cv2.imwrite(os.path.join(imdir, "output/ps1-5-a-2.png"), img_edges)

    H = ps1_5.hough_circles_acc(img_edges, 20)
    H_peaks = np.reshape(H, (1, ) + H.shape)

    peaks = ps1_2.hough_peaks(H_peaks, threshold=140, nhoodsize=10)[:, 1:]

    _ = ps1_5.hough_circles_draw(img_circles, os.path.join(
        imdir, "output/ps1-5-a-3.png"), peaks, 20)

    centers, radii = ps1_utils.findCircles(img_edges, radius_range=[20, 50])

    for center, radius in zip(centers, radii):
        cv2.circle(np.dstack((img_gray, img_gray, img_gray)),
                   center[::-1], radius, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(imdir, "output/ps1-5-b-1.png"), img_circles)


def problem6(imdir):
    img = cv2.imread(os.path.join(
        imdir, "input/ps1-input2.png"), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_smooth = cv2.GaussianBlur(img_gray, (7, 7), 3)
    img_lines = np.dstack((img_smooth, img_smooth, img_smooth))
    img_filtered_lines = np.dstack((img_smooth, img_smooth, img_smooth))

    img_edges = ps1_1.canny(img_smooth, low=50, high=100)

    H, thetas, rhos = ps1_2.hough_lines_acc(img_edges)
    H = cv2.normalize(H, H, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    H_peaks = np.reshape(H, (1, ) + H.shape)
    peaks = ps1_2.hough_peaks(H_peaks, threshold=120, nhoodsize=50)[:, 1:]

    _ = ps1_2.hough_lines_draw(img_lines,
                               os.path.join(imdir, "output/ps1-6-a-1.png"),
                               peaks, rhos, thetas)

    peaks = util.filter_lines(peaks, thetas, rhos, 5, 50)

    _ = ps1_2.hough_lines_draw(img_filtered_lines,
                               os.path.join(imdir, "output/ps1-6-c-1.png"),
                               peaks, rhos, thetas)


def problem7(imdir):
    img = cv2.imread(os.path.join(
        imdir, "input/ps1-input2.png"), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_erode = cv2.erode(img_gray, np.ones((3, 3), np.uint8), 1)
    img_smooth = cv2.blur(img_erode, (3, 3))
    img_circles = np.dstack((img_smooth, img_smooth, img_smooth))

    img_edges = ps1_1.canny(img_smooth)
    centers, radii = ps1_utils.findCircles(img_edges,
                                           radius_range=[20, 40],
                                           threshold=135,
                                           nhoodsize=20)

    for center, radius in zip(centers, radii):
        cv2.circle(img_circles, center[::-1], radius, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(imdir, "output/ps1-7-a-1.png"), img_circles)


def problem8(imdir):
    img = cv2.imread(os.path.join(
        imdir, "input/ps1-input3.png"), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_erode = cv2.erode(img_gray, np.ones((3, 3), np.uint8), 1)
    img_smooth = cv2.GaussianBlur(img_erode, (3, 3), 2)
    img_lines = np.dstack((img_smooth, img_smooth, img_smooth))

    img_edges = ps1_1.canny(img_smooth, low=40, high=80)

    H, thetas, rhos = ps1_2.hough_lines_acc(img_edges)
    H = cv2.normalize(H, H, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    H_peaks = np.reshape(H, (1, ) + H.shape)
    peaks = ps1_2.hough_peaks(H_peaks, numpeaks=40,
                              threshold=105, nhoodsize=40)[:, 1:]

    peaks = ps1_utils.filter_lines(peaks, thetas, rhos, 3, 24)

    _ = ps1_2.hough_lines_draw(img_lines,
                               os.path.join(imdir, "output/ps1-8-a-1.png"),
                               peaks, rhos, np.deg2rad(thetas))


if __name__ == "__main__":

    imdir = Path(__file__).parent

    # problem2(imdir)
    # problem3(imdir)
    # problem4(imdir)
    # problem5(imdir)
    # problem6(imdir)
    # problem7(imdir)
    problem8(imdir)
