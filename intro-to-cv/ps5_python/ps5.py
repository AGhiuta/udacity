import glob

from optic_flow_utils import *
from draw_utils import *
from image_pyr_utils import *
from warp_utils import *


def ps5_1_a():
    frame1 = cv2.imread("input/TestSeq/Shift0.png", cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread("input/TestSeq/ShiftR2.png", cv2.IMREAD_GRAYSCALE)
    frame3 = cv2.imread("input/TestSeq/ShiftR5U5.png", cv2.IMREAD_GRAYSCALE)

    op_flow_12 = lk_optic_flow(frame1, frame2, wsize=15)
    op_flow_13 = lk_optic_flow(frame1, frame3, wsize=15)

    disp_optic_flow(frame1, op_flow_12, "output/ps5-1-a-1.png")
    disp_optic_flow(frame1, op_flow_13, "output/ps5-1-a-2.png")


def ps5_1_b():
    frames = [cv2.imread(
        "input/TestSeq/Shift{}.png".format(fname),
        cv2.IMREAD_GRAYSCALE) for fname in ["0", "R10", "R20", "R40"]]

    op_flows = [lk_optic_flow(frames[0], frame1, wsize=15)
                for frame1 in frames[1:]]

    for i, op_flow in enumerate(op_flows):
        disp_optic_flow(frames[0], op_flow,
                        "output/ps5-1-b-{}.png".format(i+1))


def ps5_2_a():
    frame = cv2.imread("input/DataSeq1/yos_img_01.jpg")
    pyr = gauss_pyr(frame)

    disp_pyr(pyr, "ps5-2-a-1.png")


def ps5_2_b():
    frame = cv2.imread("input/DataSeq1/yos_img_01.jpg")
    pyr = laplace_pyr(frame)

    disp_pyr(pyr, "ps5-2-b-1.png")


def ps5_3_a(dname, output, num_levels=4, wsize=15):
    fnames = sorted(glob.glob("input/{}/*".format(dname)))
    frames = [cv2.imread(fname, cv2.IMREAD_GRAYSCALE) for fname in fnames]
    pyrs = [gauss_pyr(frame, num_levels=num_levels)[-1] for frame in frames]
    op_flows = []

    for i in range(len(pyrs) - 1):
        op_flows.append(-lk_optic_flow(pyrs[i], pyrs[i+1], wsize=wsize))

    warps = [warp_flow(frame, op_flow)
             for (frame, op_flow) in zip(pyrs[1:], op_flows)]
    diff_imgs = [cv2.cvtColor(cv2.subtract(
        pyrs[i], warps[i]), cv2.COLOR_GRAY2BGR) for i in range(len(warps))]

    tokens = output.split('.')

    try:
        pos = len(tokens[0]) - tokens[0][::-1].index('-')
        img_id = int(tokens[0][pos:]) + 1
    except:
        pos = len(tokens[0])
        img_id = 1

    disp_optic_flows(pyrs[:-1], op_flows, output)
    disp_imgs(diff_imgs, '.'.join(
        ["{}{}".format(tokens[0][:pos], img_id), tokens[1]]))


def ps5_4_a(fnames, output, num_levels=4, wsize=15):
    frames = [cv2.imread(fname, cv2.IMREAD_GRAYSCALE) for fname in fnames]
    L = frames[0]
    op_flows = []
    diff_imgs = []

    for R in frames[1:]:
        op_flows.append(hlk_optic_flow(
            [L, R], num_levels, wsize))
        wL = warp_flow(L, op_flows[-1])
        diff_imgs.append(cv2.cvtColor(cv2.subtract(wL, R), cv2.COLOR_GRAY2BGR))

    disp_optic_flows([L]*len(op_flows), op_flows, output[0], scale=50)
    disp_imgs(diff_imgs, output[1])


def ps5_5_a(fnames, output, num_levels=4, wsize=15):
    ps5_4_a(fnames, output, num_levels, wsize)


if __name__ == "__main__":
    # ps5_1_a()
    # ps5_1_b()
    # ps5_2_a()
    # ps5_2_b()
    # ps5_3_a("DataSeq1", "output/ps5-3-a-1.png", num_levels=1)
    # ps5_3_a("DataSeq2", "output/ps5-3-a-3.png", num_levels=3, wsize=7)
    # ps5_4_a(["input/TestSeq/Shift{}.png".format(fname)
    #          for fname in ['0', "R10", "R20", "R40"]],
    #         ["output/ps5-4-a-{}.png".format(i) for i in range(1, 3)],
    #         num_levels=3, wsize=15)
    # ps5_4_a(["input/DataSeq1/yos_img_0{}.jpg".format(i)
    #          for i in range(1, 4)],
    #         ["output/ps5-4-b-{}.png".format(i) for i in range(1, 3)],
    #         num_levels=1, wsize=15)
    # ps5_4_a(["input/DataSeq2/{}.png".format(i) for i in range(3)],
    #         ["output/ps5-4-c-{}.png".format(i) for i in range(1, 3)],
    #         num_levels=3, wsize=15)
    ps5_5_a(["input/Juggle/{}.png".format(i) for i in range(3)],
            ["output/ps5-5-a-{}.png".format(i) for i in range(1, 3)],
            num_levels=1, wsize=15)
