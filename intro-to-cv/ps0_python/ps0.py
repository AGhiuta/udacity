"""
Problem Set 0: Images as Functions
"""

import cv2
import os
import numpy as np
from pathlib import Path


def swapChannels(img, chan1, chan2):
    img_swapped = img.copy()
    img_swapped[:, :, chan1], img_swapped[:, :, chan2] = \
        img_swapped[:, :, chan2], img_swapped[:, :, chan1]

    return img_swapped


def replaceRegion(img1, img2, region_height, region_width):
    img1_height, img1_width = img1.shape
    img1_center_y = (img1_height - region_height) // 2
    img1_center_x = (img1_width - region_width) // 2

    img2_height, img2_width = img2.shape
    img2_center_y = (img2_height - region_height) // 2
    img2_center_x = (img2_width - region_width) // 2

    img2_replaced = img2.copy()
    img2_replaced[img2_center_y:img2_center_y+region_height,
                  img2_center_x:img2_center_x+region_width] = \
        img1[img1_center_y:img1_center_y+region_height,
             img1_center_x:img1_center_x+region_width]

    return img2_replaced


if __name__ == "__main__":

    imdir = Path(__file__).parent

    # Swap red and blue pixels of image 1
    img1 = cv2.imread(os.path.join(
        imdir, "output/ps0-1-a-1.png"), cv2.IMREAD_COLOR)
    img1_swapped = swapChannels(img1, 0, 2)

    cv2.imwrite(os.path.join(imdir, "output/ps0-2-a-1.png"), img1_swapped)

    # Create a monochrome image by selecting the green channel of image 1
    img1_green = img1[:, :, 1]
    cv2.imwrite(os.path.join(imdir, "output/ps0-2-b-1.png"), img1_green)

    # Create a monochrome image by selecting the red channel of image 1
    img1_red = img1[:, :, 2]
    cv2.imwrite(os.path.join(imdir, "output/ps0-2-c-1.png"), img1_red)

    # Take the center square region of 100x100 pixels of monochrome version of
    # image 1 and insert them into the center of monochrome version of image 2
    img2 = cv2.imread(os.path.join(
        imdir, "output/ps0-1-a-2.png"), cv2.IMREAD_COLOR)
    img2_green = img2[:, :, 1]
    img2_replaced = replaceRegion(img1_green, img2_green, 100, 100)

    cv2.imwrite(os.path.join(imdir, "output/ps0-3-a-1.png"), img2_replaced)

    # The min and max of the pixel values of img1_green
    min_pixel, max_pixel = img1_green.min(), img1_green.max()
    print('Min of the pixel value of img1_green: {}'.format(min_pixel))
    print('Max of the pixel value of img1_green: {}'.format(max_pixel))

    # The mean of the pixel values of img1_green
    mean_pixel = img1_green.mean()
    print('Mean of the pixel value of img1_green: {}'.format(mean_pixel))

    # The stddev of the pixel values of img1_green
    stddev = img1_green.std()
    print('Stddev of the pixel value of img1_green: {}'.format(stddev))

    # Subtract mean, divide by stddev, multiply by 10 then add the mean back
    img1_green_norm = img1_green.copy()
    img1_green_norm = cv2.add(cv2.multiply(cv2.divide(cv2.subtract(
        img1_green_norm, mean_pixel), stddev), 10), mean_pixel)

    cv2.imwrite(os.path.join(imdir, "output/ps0-4-b-1.png"), img1_green_norm)

    # Shift img1_green to the left by 2 pixels
    img1_green_shifted = np.zeros_like(img1_green_norm)
    img1_green_shifted[:, :-2] = img1_green_norm[:, 2:]

    cv2.imwrite(os.path.join(imdir, "output/ps0-4-c-1.png"),
                img1_green_shifted)

    # Subtract the shifted version of img1_green from the original
    img1_green_diff = img1_green_norm - img1_green_shifted

    cv2.imwrite(os.path.join(imdir, "output/ps0-4-d-1.png"),
                img1_green_diff)

    # Add Gaussian noise to the pixels in the green channel of original image 1
    img1_noise = img1.copy()
    noise = np.zeros_like(img1_green)
    mean, std = 0, 30
    cv2.randn(noise, mean, std)
    img1_noise[:, :, 1] += noise

    cv2.imwrite(os.path.join(imdir, "output/ps0-5-a-1.png"), img1_noise)

    # Add that amount of noise to the blue channel
    img1_blue_noise = img1.copy()
    img1_blue_noise[:, :, 0] += noise

    cv2.imwrite(os.path.join(imdir, "output/ps0-5-b-1.png"), img1_blue_noise)
