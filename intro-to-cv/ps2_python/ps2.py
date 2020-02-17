# ps2
from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr
import os
import numpy as np
import cv2


def ps2_1():
    # Read images
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), cv2.IMREAD_GRAYSCALE)
    L = cv2.normalize(L, L, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    R = cv2.normalize(R, R, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = disparity_ssd(L, R)
    D_R = disparity_ssd(R, L)

    # TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
    # Note: They may need to be scaled/shifted before saving to show results properly

    D_L = cv2.normalize(D_L, D_L, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite("output/ps2-1-a-1.png", D_L)
    cv2.imwrite("output/ps2-1-a-2.png", D_R)


# TODO: Rest of your code here
def ps2_2():
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE)

    # Downsample
    L = cv2.pyrDown(L)
    R = cv2.pyrDown(R)
    L = cv2.normalize(L, L, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    R = cv2.normalize(R, R, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = np.abs(disparity_ssd(L, R, tplSize=[7, 7], disparity=100))
    D_R = np.abs(disparity_ssd(R, L, tplSize=[7, 7], disparity=100))

    D_L = cv2.normalize(D_L, D_L, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Upsample
    D_L = cv2.pyrUp(D_L)
    D_R = cv2.pyrUp(D_R)

    cv2.imwrite("output/ps2-2-a-1.png", D_L)
    cv2.imwrite("output/ps2-2-a-2.png", D_R)


def ps2_3():
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE)

    # Downsample
    L = cv2.pyrDown(L)
    R = cv2.pyrDown(R)
    L = cv2.normalize(L, L, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    R = cv2.normalize(R, R, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Add Gaussian noise to the left image
    noise = np.empty(L.shape, dtype=np.float32)
    cv2.randn(noise, mean=0, stddev=0.05)
    L_noise = L + noise
    L_noise = cv2.normalize(L_noise, L_noise, 0, 1,
                            cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Increase the contrast of the left image by 10%
    L_contrast = L * 1.1
    L_contrast = cv2.normalize(L_contrast, L_contrast, 0, 1,
                               cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L_noise = np.abs(disparity_ssd(
        L_noise, R, tplSize=[7, 7], disparity=100))
    D_R_noise = np.abs(disparity_ssd(
        R, L_noise, tplSize=[7, 7], disparity=100))

    D_L_contrast = np.abs(disparity_ssd(
        L_contrast, R, tplSize=[7, 7], disparity=100))
    D_R_contrast = np.abs(disparity_ssd(
        R, L_contrast, tplSize=[7, 7], disparity=100))

    D_L_noise = cv2.normalize(
        D_L_noise, D_L_noise, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_noise = cv2.normalize(
        D_R_noise, D_R_noise, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    D_L_contrast = cv2.normalize(
        D_L_contrast, D_L_contrast, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_contrast = cv2.normalize(
        D_R_contrast, D_R_contrast, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Upsample
    D_L_noise = cv2.pyrUp(D_L_noise)
    D_R_noise = cv2.pyrUp(D_R_noise)
    D_L_contrast = cv2.pyrUp(D_L_contrast)
    D_R_contrast = cv2.pyrUp(D_R_contrast)

    cv2.imwrite("output/ps2-3-a-1.png", D_L_noise)
    cv2.imwrite("output/ps2-3-a-2.png", D_R_noise)
    cv2.imwrite("output/ps2-3-b-1.png", D_L_contrast)
    cv2.imwrite("output/ps2-3-b-2.png", D_R_contrast)


def ps2_4():
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE)

    # Downsample
    L = cv2.pyrDown(L)
    R = cv2.pyrDown(R)
    L = cv2.normalize(L, L, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    R = cv2.normalize(R, R, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = np.abs(disparity_ncorr(L, R, tplSize=[7, 7], disparity=100))
    D_R = np.abs(disparity_ncorr(R, L, tplSize=[7, 7], disparity=100))

    D_L = cv2.normalize(D_L, D_L, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Upsample
    D_L = cv2.pyrUp(D_L)
    D_R = cv2.pyrUp(D_R)

    cv2.imwrite("output/ps2-4-a-1.png", D_L)
    cv2.imwrite("output/ps2-4-a-2.png", D_R)

    # Add Gaussian noise to the left image
    noise = np.empty(L.shape, dtype=np.float32)
    cv2.randn(noise, mean=0, stddev=0.05)
    L_noise = L + noise
    L_noise = cv2.normalize(L_noise, L_noise, 0, 1,
                            cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Increase the contrast of the left image by 10%
    L_contrast = L * 1.1
    L_contrast = cv2.normalize(L_contrast, L_contrast, 0, 1,
                               cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L_noise = np.abs(disparity_ncorr(
        L_noise, R, tplSize=[7, 7], disparity=100))
    D_R_noise = np.abs(disparity_ncorr(
        R, L_noise, tplSize=[7, 7], disparity=100))

    D_L_contrast = np.abs(disparity_ncorr(
        L_contrast, R, tplSize=[7, 7], disparity=100))
    D_R_contrast = np.abs(disparity_ncorr(
        R, L_contrast, tplSize=[7, 7], disparity=100))

    D_L_noise = cv2.normalize(
        D_L_noise, D_L_noise, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_noise = cv2.normalize(
        D_R_noise, D_R_noise, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    D_L_contrast = cv2.normalize(
        D_L_contrast, D_L_contrast, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R_contrast = cv2.normalize(
        D_R_contrast, D_R_contrast, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Upsample
    D_L_noise = cv2.pyrUp(D_L_noise)
    D_R_noise = cv2.pyrUp(D_R_noise)
    D_L_contrast = cv2.pyrUp(D_L_contrast)
    D_R_contrast = cv2.pyrUp(D_R_contrast)

    cv2.imwrite("output/ps2-4-b-1.png", D_L_noise)
    cv2.imwrite("output/ps2-4-b-2.png", D_R_noise)
    cv2.imwrite("output/ps2-4-b-3.png", D_L_contrast)
    cv2.imwrite("output/ps2-4-b-4.png", D_R_contrast)


def ps2_5():
    L = cv2.imread(os.path.join('input', 'pair2-L.png'), cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(os.path.join('input', 'pair2-R.png'), cv2.IMREAD_GRAYSCALE)

    # Downsample
    L = cv2.pyrDown(L)
    R = cv2.pyrDown(R)

    # Normalize both images
    L = cv2.normalize(L, L, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    R = cv2.normalize(R, R, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = np.abs(disparity_ncorr(L, R, tplSize=[11, 11], disparity=100))
    D_R = np.abs(disparity_ncorr(R, L, tplSize=[11, 11], disparity=100))

    D_L = cv2.normalize(D_L, D_L, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    D_R = cv2.normalize(D_R, D_R, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Upsample
    D_L = cv2.pyrUp(D_L)
    D_R = cv2.pyrUp(D_R)

    cv2.imwrite("output/ps2-5-a-1.png", D_L)
    cv2.imwrite("output/ps2-5-a-2.png", D_R)


if __name__ == "__main__":
    # ps2_1()
    # ps2_2()
    # ps2_3()
    # ps2_4()
    ps2_5()
