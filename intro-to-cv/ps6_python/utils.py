import cv2
import numpy as np
import sys

from pf_tracker import PFTracker
from msl_tracker import MSLTracker


def pf_tracking(filename, save_frames=[], pref="1-a", save_video=False,
                num_particles=100, state_dims=2, std_control=10, std_MSE=10,
                alpha=0):
    vname = '_'.join(filename.split('_')[:2])
    cap = cv2.VideoCapture("input/{}.avi".format(vname))
    ret, frame0 = cap.read()

    assert ret == True

    frame0_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    x, y, w, h = map(lambda x: int(float(x)), open(
        "input/{}.txt".format(filename)).read().split())
    model = frame0_gray[y:y+h, x:x+w]

    cv2.imwrite("output/ps6-{}-1.png".format(pref), frame0[y:y+h, x:x+w])

    search_space = frame0_gray.shape
    pf_tracker = PFTracker(
        model, search_space[::-1], num_particles, state_dims, std_control,
        std_MSE, alpha)

    pf_tracker.visualize(frame0)

    if save_video:
        out = cv2.VideoWriter("output/{}.avi".format(filename),
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              cap.get(cv2.CAP_PROP_FPS), frame0.shape[:2][::-1])
        out.write(frame0)

    frame_count, frames_saved = 1, 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pf_tracker.update(frame_gray)
        pf_tracker.visualize(frame)

        if frame_count in save_frames:
            cv2.imwrite(
                "output/ps6-{}-{}.png".format(pref, frames_saved+2), frame)
            frames_saved += 1

        if save_video:
            out.write(frame)

    cap.release()

    if save_video:
        out.release()


def msl_tracking(filename, save_frames=[], pref="3-a", save_video=False,
                 num_particles=100, state_dims=2, std_control=10, std_CHI=10,
                 num_bins=8, color_space="BGR", alpha=0):

    vname = '_'.join(filename.split('_')[:2])
    cap = cv2.VideoCapture("input/{}.avi".format(vname))
    ret, frame0 = cap.read()

    assert ret == True

    x, y, w, h = map(lambda x: int(float(x)), open(
        "input/{}.txt".format(filename)).read().split())

    model = frame0[y:y+h, x:x+w]

    if color_space == "GRAY":
        model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
    elif color_space == "HSV":
        model = cv2.cvtColor(model, cv2.COLOR_BGR2HSV)

    cv2.imwrite("output/ps6-{}-1.png".format(pref), frame0[y:y+h, x:x+w])

    search_space = frame0.shape[:2]
    msl_tracker = MSLTracker(
        model, search_space[::-1], num_particles, state_dims, std_control,
        std_CHI, num_bins, color_space, alpha)

    msl_tracker.visualize(frame0)

    if save_video:
        out = cv2.VideoWriter("output/{}_msl.avi".format(filename),
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              cap.get(cv2.CAP_PROP_FPS), frame0.shape[:2][::-1])
        out.write(frame0)

    frame_count, frames_saved = 1, 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        msl_tracker.update(frame)
        msl_tracker.visualize(frame)

        if frame_count in save_frames:
            cv2.imwrite(
                "output/ps6-{}-{}.png".format(pref, frames_saved+2), frame)
            frames_saved += 1

        if save_video:
            out.write(frame)

    cap.release()

    if save_video:
        out.release()
