import cv2
import numpy as np

from grad_utils import *
from draw_utils import *


def ps4_1_a(img_names_list, ksize=3, normalize=True, save=True):
    grads_x = []
    grads_y = []
    imgs = []

    for i, img_name in enumerate(img_names_list):
        img = cv2.imread("input/{}".format(img_name), cv2.IMREAD_GRAYSCALE)
        grad_x = compute_x_grad(img, ksize=ksize, normalize=normalize)
        grad_y = compute_y_grad(img, ksize=ksize, normalize=normalize)

        grads_x.append(grad_x)
        grads_y.append(grad_y)
        imgs.append(img)

        img_grad = np.hstack((grad_x, grad_y))

        if save:
            cv2.imwrite("output/ps4-1-a-{}.png".format(i+1), img_grad)

    return imgs, grads_x, grads_y


def ps4_1_b(img_names_list, wsize=5, alpha=0.04, normalize=True, save=True):
    # compute the x and y gradients for each image
    imgs, grads_x, grads_y = ps4_1_a(
        img_names_list, normalize=False, save=False)

    # compute the weights matrix
    w = np.zeros((wsize, wsize), dtype=np.float)
    w[wsize//2, wsize//2] = 1
    w = cv2.GaussianBlur(w, (wsize, wsize), 1)

    # store the results
    Rs = []

    for i, (img, Ix, Iy) in enumerate(zip(imgs, grads_x, grads_y)):
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy

        height, width = img.shape
        R = np.zeros(img.shape, dtype=np.float)

        for row in range(wsize//2, height-wsize//2):
            min_row = row - wsize//2
            max_row = min_row + wsize

            for col in range(wsize//2, width-wsize//2):
                min_col = col - wsize//2
                max_col = min_col + wsize

                M = np.array([
                    np.sum(w * Ixx[min_row:max_row, min_col:max_col]),
                    np.sum(w * Ixy[min_row:max_row, min_col:max_col]),
                    np.sum(w * Ixy[min_row:max_row, min_col:max_col]),
                    np.sum(w * Iyy[min_row:max_row, min_col:max_col])
                ]).reshape((2, 2))

                R[row, col] = np.linalg.det(M) - alpha * np.trace(M) ** 2

        if normalize:
            R = cv2.normalize(R, R, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        if save:
            cv2.imwrite("output/ps4-1-b-{}.png".format(i+1), R)

        Rs.append(R)

    return imgs, Rs


def ps4_1_c(img_names_list, threshold=0.25, nms_size=5, save=True):
    imgs, Rs = ps4_1_b(img_names_list, normalize=False, save=False)
    corners = [[] for _ in range(len(imgs))]

    for i, (img, R) in enumerate(zip(imgs, Rs)):
        R = cv2.normalize(R, R, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
        R[np.where(R < threshold)] = 0
        rows, cols = np.nonzero(R)
        height, width = img.shape

        for row, col in zip(rows, cols):
            min_row = max(row - nms_size // 2, 0)
            max_row = min(min_row + nms_size, height)
            min_col = max(col - nms_size // 2, 0)
            max_col = min(min_col + nms_size, width)
            if R[row, col] == np.max(R[min_row:max_row, min_col:max_col]):
                corners[i].append((row, col))

        if save:
            img_harris = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            for corner in corners[i]:
                cv2.circle(img_harris, corner[::-1], 1, (0, 255, 0), -1)

            cv2.imwrite("output/ps4-1-c-{}.png".format(i+1), img_harris)

    return imgs, corners


def ps4_2_a(img_names_list, save=True):
    imgs, corners = ps4_1_c(img_names_list, save=False)
    kps = [[] for _ in range(len(imgs))]
    kp_imgs = []

    for i, img in enumerate(imgs):
        grad_x = compute_x_grad(img, ksize=3, normalize=False)
        grad_y = compute_y_grad(img, ksize=3, normalize=False)
        angle = compute_angle(grad_x, grad_y)

        for (row, col) in corners[i]:
            kps[i].append(cv2.KeyPoint(
                col, row, _size=10, _angle=np.rad2deg(angle[row, col])))

        if save:
            kp_imgs.append(cv2.drawKeypoints(
                img, kps[i], None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

            if i % 2 == 1:
                cv2.imwrite("output/ps4-2-a-{}.png".format((i >> 1)+1),
                            np.hstack(kp_imgs[-2:]))

    return imgs, kps


def ps4_2_b(img_names_list, save=True):
    imgs, kps = ps4_2_a(img_names_list, save=False)
    descs = []
    matches = []
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(crossCheck=True)

    for i, (img, kp) in enumerate(zip(imgs, kps)):
        descs.append(sift.compute(img, kp)[1])

        if i % 2 == 1:
            matches.append(bf.match(descs[-2], descs[-1]))

            if save:

                img_matches = draw_matches(
                    imgs[i-1], kps[i-1], imgs[i], kps[i], matches[-1])

                cv2.imwrite("output/ps4-2-b-{}.png".format((i >> 1)+1),
                            img_matches)

    return imgs, kps, matches


def ps4_3_a(img_names_list, tol=5, num_steps=10, save=True):
    imgs, kps, matches = ps4_2_b(img_names_list, save=False)

    for i, matches_i in enumerate(matches):
        best_consensus_set = []
        best_t = None

        for _ in range(num_steps):
            idx = np.random.randint(len(matches_i))
            pt1 = np.array(kps[i << 1][matches_i[idx].queryIdx].pt)
            pt2 = np.array(kps[(i << 1)+1][matches_i[idx].trainIdx].pt)
            T = pt2 - pt1
            consensus_set = []

            for j, match in enumerate(matches_i):
                pt1 = np.array(kps[i << 1][match.queryIdx].pt)
                pt2 = np.array(kps[(i << 1)+1][match.trainIdx].pt)
                pt_trans = T + pt1

                if np.linalg.norm(pt_trans - pt2) < tol:
                    consensus_set.append(j)

            if len(consensus_set) > len(best_consensus_set):
                best_consensus_set = consensus_set
                best_t = T

        if save:
            best_matches = np.array(matches_i)[best_consensus_set]
            img_matches = draw_matches(
                imgs[i << 1], kps[i << 1], imgs[(i << 1)+1],
                kps[(i << 1)+1], best_matches)

            cv2.imwrite("output/ps4-3-a-{}.png".format(i+1),
                        img_matches)

            print(
                "Percentage of matches in the biggest consensus set: {:.2f}"
                .format(100.0 * len(best_matches) / len(matches_i)))
            print("Translation vector: {}".format(best_t))


def ps4_3_b(img_names_list, tol=5, num_steps=10, save=True):
    imgs, kps, matches = ps4_2_b(img_names_list, save=False)
    best_Ss = []

    for i, matches_i in enumerate(matches):
        best_consensus_set = []
        best_S = None

        for _ in range(num_steps):
            idx = np.random.randint(len(matches_i), size=2)
            pt11 = kps[i << 1][matches_i[idx[0]].queryIdx].pt
            pt12 = kps[(i << 1)+1][matches_i[idx[0]].trainIdx].pt
            pt21 = kps[i << 1][matches_i[idx[1]].queryIdx].pt
            pt22 = kps[(i << 1)+1][matches_i[idx[1]].trainIdx].pt

            A = np.array([[pt11[0], -pt11[1], 1, 0],
                          [pt11[1], pt11[0], 0, 1],
                          [pt21[0], -pt21[1], 1, 0],
                          [pt21[1], pt21[0], 0, 1]])
            b = np.array([pt12[0], pt12[1], pt22[0], pt22[1]])

            S, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            S = np.array([[S[0], -S[1], S[2]],
                          [S[1], S[0], S[3]]])

            consensus_set = []

            for j, match in enumerate(matches_i):
                pt1 = np.insert(np.array(kps[i << 1][match.queryIdx].pt), 2, 1)
                pt2 = np.array(kps[(i << 1)+1][match.trainIdx].pt)
                pt_sim = np.dot(S, pt1)

                if np.linalg.norm(pt_sim - pt2) < tol:
                    consensus_set.append(j)

            if len(consensus_set) > len(best_consensus_set):
                best_consensus_set = consensus_set
                best_S = S

        if save:
            best_matches = np.array(matches_i)[best_consensus_set]
            img_matches = draw_matches(
                imgs[i << 1], kps[i << 1], imgs[(i << 1)+1],
                kps[(i << 1)+1], best_matches)

            cv2.imwrite("output/ps4-3-b-{}.png".format(i+1),
                        img_matches)

            print(
                "Percentage of matches in the biggest consensus set: {:.2f}"
                .format(100.0 * len(best_matches) / len(matches_i)))
            print("Transform matrix: {}".format(best_S))

        best_Ss.append(best_S)

    return imgs, best_Ss


def ps4_3_c(img_names_list, tol=5, num_steps=10, save=True):
    imgs, kps, matches = ps4_2_b(img_names_list, save=False)
    best_SAs = []

    for i, matches_i in enumerate(matches):
        best_consensus_set = []
        best_SA = None

        for _ in range(num_steps):
            idx = np.random.randint(len(matches_i), size=3)
            pt11 = kps[i << 1][matches_i[idx[0]].queryIdx].pt
            pt12 = kps[(i << 1)+1][matches_i[idx[0]].trainIdx].pt
            pt21 = kps[i << 1][matches_i[idx[1]].queryIdx].pt
            pt22 = kps[(i << 1)+1][matches_i[idx[1]].trainIdx].pt
            pt31 = kps[i << 1][matches_i[idx[2]].queryIdx].pt
            pt32 = kps[(i << 1)+1][matches_i[idx[2]].trainIdx].pt

            A = np.array([[pt11[0], pt11[1], 1, 0, 0, 0],
                          [0, 0, 0, pt11[0], pt11[1], 1],
                          [pt21[0], pt21[1], 1, 0, 0, 0],
                          [0, 0, 0, pt21[0], pt21[1], 1],
                          [pt31[0], pt31[1], 1, 0, 0, 0],
                          [0, 0, 0, pt31[0], pt31[1], 1]])
            b = np.array([pt12[0], pt12[1], pt22[0],
                          pt22[1], pt32[0], pt32[1]])

            SA, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            SA = np.reshape(SA, (2, 3))

            consensus_set = []

            for j, match in enumerate(matches_i):
                pt1 = np.insert(np.array(kps[i << 1][match.queryIdx].pt), 2, 1)
                pt2 = np.array(kps[(i << 1)+1][match.trainIdx].pt)
                pt_sim = np.dot(SA, pt1)

                if np.linalg.norm(pt_sim - pt2) < tol:
                    consensus_set.append(j)

            if len(consensus_set) > len(best_consensus_set):
                best_consensus_set = consensus_set
                best_SA = SA

        if save:
            best_matches = np.array(matches_i)[best_consensus_set]
            img_matches = draw_matches(
                imgs[i << 1], kps[i << 1], imgs[(i << 1)+1],
                kps[(i << 1)+1], best_matches)

            cv2.imwrite("output/ps4-3-c-{}.png".format(i+1),
                        img_matches)

            print(
                "Percentage of matches in the biggest consensus set: {:.2f}"
                .format(100.0 * len(best_matches) / len(matches_i)))
            print("Transform matrix: {}".format(best_SA))

        best_SAs.append(best_SA)

    return imgs, best_SAs


def ps4_3_d(img_names_list, save=True):
    imgs, Ts = ps4_3_b(img_names_list, save=False)

    for i, T in enumerate(Ts):
        src = imgs[i << 1]
        dst = imgs[(i << 1) + 1]
        warped_dst = cv2.warpAffine(dst, T, dst.shape[1::-1],
                                    flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        if save:
            overlay = cv2.cvtColor(np.zeros_like(src), cv2.COLOR_GRAY2BGR)
            overlay[:, :, 1] = src
            overlay[:, :, 2] = warped_dst

            cv2.imwrite("output/ps4-3-d-{}.png".format((i << 2)+1), warped_dst)
            cv2.imwrite("output/ps4-3-d-{}.png".format((i << 2)+2), overlay)


def ps4_3_e(img_names_list, save=True):
    imgs, Ts = ps4_3_c(img_names_list, save=False)

    for i, T in enumerate(Ts):
        src = imgs[i << 1]
        dst = imgs[(i << 1) + 1]
        warped_dst = cv2.warpAffine(dst, T, dst.shape[1::-1],
                                    flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

        if save:
            overlay = cv2.cvtColor(np.zeros_like(src), cv2.COLOR_GRAY2BGR)
            overlay[:, :, 1] = src
            overlay[:, :, 2] = warped_dst

            cv2.imwrite("output/ps4-3-e-{}.png".format((i << 2)+1), warped_dst)
            cv2.imwrite("output/ps4-3-e-{}.png".format((i << 2)+2), overlay)


if __name__ == "__main__":
    img_names_list = ["transA.jpg", "transB.jpg", "simA.jpg", "simB.jpg", ]

    # _, _, _ = ps4_1_a(img_names_list[0:3:2])
    # _ = ps4_1_b(img_names_list)
    # _, _ = ps4_1_c(img_names_list)
    # _, _ = ps4_2_a(img_names_list)
    # _, _ = ps4_2_b(img_names_list)
    # ps4_3_a(img_names_list[:2])
    # ps4_3_b(img_names_list[2:])
    # ps4_3_c(img_names_list[2:])
    # ps4_3_d(img_names_list[2:])
    ps4_3_e(img_names_list[2:])
