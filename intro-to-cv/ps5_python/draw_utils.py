import cv2
import matplotlib.pyplot as plt
import numpy as np


def disp_optic_flow(frame, op_flow, fname, scale=50):
    h, w = frame.shape[:2]
    x, y = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))

    plt.figure()
    _ = plt.imshow(frame, cmap="gray", interpolation="bicubic")
    step = h // scale

    plt.quiver(x[::step, ::step], y[::step, ::step],
               op_flow[::step, ::step, 0], op_flow[::step, ::step, 1],
               color="r", pivot="middle", headwidth=2, headlength=3)
    plt.axis("off")

    plt.savefig(fname, bbox_inches="tight", pad_inches=0)


def disp_optic_flows(frames, op_flows, fname, scale=30):
    nframes = len(frames)
    nrows = int(nframes**0.5)
    ncols = int(np.ceil(nframes / nrows))

    _, axs = plt.subplots(nrows, ncols, figsize=(nrows*10, ncols*10))

    if axs.ndim == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, (frame, op_flow) in enumerate(zip(frames, op_flows)):
        row, col = i // ncols, i % ncols
        h, w = frame.shape[:2]
        x, y = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
        step = h // scale

        axs[row, col].imshow(frame, cmap="gray", interpolation="bicubic")
        axs[row, col].quiver(x[::step, ::step], y[::step, ::step],
                             op_flow[::step, ::step, 0],
                             op_flow[::step, ::step, 1],
                             color="r", pivot="middle",
                             headwidth=2, headlength=3)
        axs[row, col].axis("off")

    plt.savefig(fname, bbox_inches="tight", pad_inches=0)


def disp_imgs(imgs, fname):
    nimgs = len(imgs)
    nrows = int(nimgs**0.5)
    ncols = int(np.ceil(nimgs / nrows))

    _, axs = plt.subplots(nrows, ncols, figsize=(nrows*10, ncols*10))

    if axs.ndim == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, img in enumerate(imgs):
        row, col = i // ncols, i % ncols

        axs[row, col].imshow(img, cmap="gray", interpolation="bicubic")
        axs[row, col].axis("off")

    plt.savefig(fname, bbox_inches="tight", pad_inches=0)


def disp_pyr(pyr, fname):
    h, w, c = pyr[0].shape
    img = np.ones((h + h//2 + 2, w, c), dtype=np.uint8)
    img *= 255
    img[:h, :w, :] = pyr[0]
    w_offset = 0

    for p in pyr[1:]:
        h_p, w_p = p.shape[:2]
        img[h+2:h+2+h_p, w_offset:w_offset+w_p, :] = p
        w_offset += w_p + 2

    cv2.imwrite("output/{}".format(fname), img)
