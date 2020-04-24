import cv2
import itertools
import numpy as np

from utils import compute_frame_diff_seq
from utils import compute_mhi_seq
from utils import compute_hu_moments
from plot_utils import plot_knn_confusion_matrix
from plot_utils import plot_confusion_matrix


def ps7_1_a(vname, frames_to_save):
    num_frames = max(frames_to_save)
    frame_diff_seq = compute_frame_diff_seq(vname, num_frames=num_frames,
                                            theta=2, blur_ksize=(55, 55),
                                            blur_sigma=0, morph_ksize=(9, 9))

    for i, frame_id in enumerate(frames_to_save, 1):
        diff = cv2.normalize(frame_diff_seq[frame_id-1], None,
                             0, 255, cv2.NORM_MINMAX)
        cv2.imwrite("output/ps7-1-a-{}.png".format(i), diff)


def ps7_1_b():
    t_end = [35, 30, 30]
    thetas = [4, 4, 4]
    taus = [30, 30, 30]
    vnames = ["input/PS7A{}P1T1.avi".format(x) for x in range(1, 4)]

    for idx in range(3):
        frame_diff_seq = compute_frame_diff_seq(vnames[idx],
                                                num_frames=t_end[idx],
                                                theta=thetas[idx],
                                                blur_ksize=(85, 85),
                                                blur_sigma=0,
                                                morph_ksize=(9, 9))
        M = compute_mhi_seq(frame_diff_seq, tau=taus[idx], t_end=t_end[idx])

        cv2.normalize(M, M, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite("output/ps7-1-b-{}.png".format(idx+1), M.astype(np.uint8))


def ps7_2_a(actions=[1, 2, 3], persons=[1, 2, 3], trials=[1, 2, 3], save_plots=True):
    t_end = [70, 40, 50]
    thetas = [3, 3, 20]
    taus = [60, 40, 50]
    µs, ηs = [], []

    for seq in itertools.product(actions, persons, trials):
        vname = "input/PS7A{}P{}T{}.avi".format(*seq)
        idx = seq[1] - 1
        binary_seq = compute_frame_diff_seq(vname,
                                            num_frames=t_end[idx],
                                            theta=thetas[idx],
                                            blur_ksize=(85, 85),
                                            blur_sigma=0,
                                            morph_ksize=(9, 9))
        MHI = compute_mhi_seq(binary_seq, tau=taus[idx], t_end=t_end[idx])
        MHI = cv2.normalize(MHI, None, 0, 255, cv2.NORM_MINMAX)

        # Compute MEI by thresholding MHI
        MEI = 255*(MHI > 0).astype(np.uint8)

        # Compute the central moments µ and the scale invariant moments η
        µ_mhi, η_mhi = compute_hu_moments(MHI)
        µ_mei, η_mei = compute_hu_moments(MEI)

        µs.append(np.append(µ_mhi, µ_mei))
        ηs.append(np.append(η_mhi, η_mei))

    labels = np.array(list(itertools.chain.from_iterable(
        itertools.repeat(action, len(persons) * len(trials))
        for action in actions)), dtype=np.int)
    µs = np.array(µs, dtype=np.float32)
    ηs = np.array(ηs, dtype=np.float32)

    if save_plots:
        _ = plot_knn_confusion_matrix(
            µs, labels, "output/ps7-2-a-1.png", "Confusion Matrix")
        _ = plot_knn_confusion_matrix(
            ηs, labels, "output/ps7-2-a-2.png", "Confusion Matrix")

    return µs, ηs, labels


def ps7_2_b():
    actions, trials = [1, 2, 3], [1, 2, 3]
    confusion_matrices = []

    for idx in range(3):
        test_person = idx + 1
        persons = np.delete([1, 2, 3], idx)
        _, X_train, y_train = ps7_2_a(
            actions, persons, trials, save_plots=False)
        _, X_test, y_test = ps7_2_a(
            actions, [test_person], trials, save_plots=False)

        confusion_matrices.append(plot_knn_confusion_matrix(
            X_train, y_train, X_test, y_test,
            "output/ps7-2-b-{}.png".format(test_person),
            "Confusion Matrix - Person {}".format(test_person)))

    confusion_matrix = np.sum(confusion_matrices, axis=0)
    plot_confusion_matrix(confusion_matrix,
                          classes=["a1", "a2", "a3"],
                          normalize=True,
                          title="Avg. Confusion Matrix",
                          fname="output/ps7-2-b-4.png",
                          cmap="Blues")


if __name__ == "__main__":
    # ps7_1_a("input/PS7A1P1T1.avi", frames_to_save=[10, 20, 30])
    # ps7_1_b()
    # ps7_2_a()
    ps7_2_b()
