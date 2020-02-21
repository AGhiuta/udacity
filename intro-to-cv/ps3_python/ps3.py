import cv2
import numpy as np

from read_file import read_file
from F_lstsq import F_lstsq
from M_lstsq import M_lstsq
from calibrate_camera import calibrate_camera
from normalize_pts2d import normalize_pts2d
from draw_epipolar_lines import draw_epipolar_lines

# np_float_formatter = "{:.4f}".format
# np.set_printoptions(formatter={'float_kind': np_float_formatter})


def ps3_1_a(printing=False):
    pts2d_norm = read_file("input/pts2d-norm-pic_a.txt")
    pts3d_norm = read_file("input/pts3d-norm.txt")

    M = M_lstsq(pts2d_norm, pts3d_norm)

    pt2d_proj = np.dot(M, np.append(pts3d_norm[-1], 1.0))
    pt2d_proj = pt2d_proj / pt2d_proj[-1]
    res = np.linalg.norm(pt2d_proj[:2] - pts2d_norm[-1])

    if printing:
        print("M = {}".format(M))
        print("3D point {} projected to {}".format(
            pts3d_norm[-1], pt2d_proj[:2]))
        print("Residual: {:.4f}".format(res))

    return M


def ps3_1_b(printing=False):
    pts2d = read_file("input/pts2d-pic_b.txt")
    pts3d = read_file("input/pts3d.txt")

    results = []

    for n in range(8, 20, 4):
        results.append(calibrate_camera(pts2d, pts3d, n))

    res_min, M_best = min(results)

    if printing:
        print("Best M: {}".format(M_best))
        print("Residual val for best M: {:.4f}".format(res_min))

    return M_best


def ps3_1_c(testing=False):
    if testing:
        M = ps3_1_a()
    else:
        M = ps3_1_b()

    Q, m, _ = np.split(M, [3, 4], axis=1)
    C = np.dot(-np.linalg.inv(Q), m)

    print("Location of the camera in " +
          "3D world coordinates: {}".format(np.squeeze(C)))


def ps3_2_a(printing=False, pts2d_a=None, pts2d_b=None):
    if pts2d_a is None:
        pts2d_a = read_file("input/pts2d-pic_a.txt")

    if pts2d_b is None:
        pts2d_b = read_file("input/pts2d-pic_b.txt")

    F = F_lstsq(pts2d_a, pts2d_b)

    if printing:
        print("F = {}".format(F))

    return F


def ps3_2_b(printing=False, pts2d_a=None, pts2d_b=None):
    F = ps3_2_a(pts2d_a=pts2d_a, pts2d_b=pts2d_b)
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(np.dot(U, np.diag(S)), V)

    if printing:
        print("F with rank 2: {}".format(F))

    return F


def ps3_2_c():
    # load images
    img_a = cv2.imread("input/pic_a.jpg", cv2.IMREAD_COLOR)
    img_b = cv2.imread("input/pic_b.jpg", cv2.IMREAD_COLOR)

    # load 2d points
    pts2d_a = read_file("input/pts2d-pic_a.txt")
    pts2d_b = read_file("input/pts2d-pic_b.txt")

    # compute the fundamental matrix
    F = ps3_2_b(pts2d_a=pts2d_a, pts2d_b=pts2d_b)

    img_a = draw_epipolar_lines(img_a, pts2d_b, F.T)
    img_b = draw_epipolar_lines(img_b, pts2d_a, F)

    cv2.imwrite('output/ps3-2-c-1.png', img_a)
    cv2.imwrite('output/ps3-2-c-2.png', img_b)


def ps3_2_d(printing=False, pts2d_a=None, pts2d_b=None):
    if pts2d_a is None:
        pts2d_a = read_file("input/pts2d-pic_a.txt")

    if pts2d_b is None:
        pts2d_b = read_file("input/pts2d-pic_b.txt")

    T_a, pts2d_a_norm = normalize_pts2d(pts2d_a)
    T_b, pts2d_b_norm = normalize_pts2d(pts2d_b)
    F_hat = ps3_2_b(pts2d_a=pts2d_a_norm, pts2d_b=pts2d_b_norm)

    if printing:
        print("T_a = {}".format(T_a))
        print("T_b = {}".format(T_b))
        print("F_hat = {}".format(F_hat))

    return T_a, T_b, F_hat


def ps3_2_e(printing=False):
    # load images
    img_a = cv2.imread("input/pic_a.jpg", cv2.IMREAD_COLOR)
    img_b = cv2.imread("input/pic_b.jpg", cv2.IMREAD_COLOR)

    # load 2d points
    pts2d_a = read_file("input/pts2d-pic_a.txt")
    pts2d_b = read_file("input/pts2d-pic_b.txt")

    T_a, T_b, F_hat = ps3_2_d(pts2d_a=pts2d_a, pts2d_b=pts2d_b)
    F = np.dot(np.dot(T_b.T, F_hat), T_a)

    img_a = draw_epipolar_lines(img_a, pts2d_b, F.T)
    img_b = draw_epipolar_lines(img_b, pts2d_a, F)

    cv2.imwrite('output/ps3-2-e-1.png', img_a)
    cv2.imwrite('output/ps3-2-e-2.png', img_b)

    if printing:
        print("New F = {}".format(F))


if __name__ == "__main__":
    # _ = ps3_1_a(True)
    # _ = ps3_1_b(True)
    ps3_1_c()
    # _ = ps3_2_a(True)
    # _ = ps3_2_b(True)
    # ps3_2_c()
    # ps3_2_d(True)
    # ps3_2_e(True)
