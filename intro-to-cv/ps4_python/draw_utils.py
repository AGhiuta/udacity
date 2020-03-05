import cv2
import numpy as np


def draw_matches(img1, kp1, img2, kp2, matches):
    # img_matches = cv2.drawMatches(
    #     img1, kp1, img2,
    #     kp2, matches, None,
    #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_matches = cv2.cvtColor(
        np.hstack((img1, img2)), cv2.COLOR_GRAY2BGR)
    offset = img1.shape[1]

    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0] + offset), int(pt2[1]))

        color = tuple(map(int, np.random.choice(
            range(256), size=3)))

        cv2.line(img_matches, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(img_matches, pt1, 4, color, 1, cv2.LINE_AA)
        cv2.circle(img_matches, pt2, 4, color, 1, cv2.LINE_AA)

    return img_matches
