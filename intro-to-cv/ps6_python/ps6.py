import cv2
import numpy as np

from utils import pf_tracking, msl_tracking


def ps6_1_a(filename, save_frames=[], pref="1-a", save_video=True):
    pf_tracking(filename, save_frames, pref, save_video=save_video)


def ps6_1_e(filename, save_frames=[], pref="1-e", save_video=True):
    pf_tracking(filename, save_frames, pref, save_video=save_video)


def ps6_2_a(filename, save_frames=[], pref="2-a", save_video=True,
            num_particles=100, std_control=10, std_MSE=20, alpha=0):
    pf_tracking(filename, save_frames, pref,
                save_video=save_video, num_particles=num_particles,
                std_control=std_control, std_MSE=std_MSE, alpha=alpha)


def ps6_2_b(filename, save_frames=[], pref="2-b", save_video=True,
            num_particles=100, std_control=10, std_MSE=20, alpha=0):
    pf_tracking(filename, save_frames, pref,
                save_video=save_video, num_particles=num_particles,
                std_control=std_control, std_MSE=std_MSE, alpha=alpha)


def ps6_3_a(filename, save_frames=[], pref="3-a", save_video=True,
            num_particles=100, std_control=20, std_CHI=10,
            num_bins=8, alpha=0):
    msl_tracking(filename, save_frames, pref,
                 save_video=save_video, num_particles=num_particles,
                 std_control=std_control, std_CHI=std_CHI,
                 num_bins=num_bins, alpha=alpha)


if __name__ == "__main__":
    # ps6_1_a("pres_debate", [28, 84, 144])
    # ps6_1_e("noisy_debate", [14, 32, 46])
    # ps6_2_a("pres_debate_hand", [15, 50, 140],
    #         num_particles=600, std_control=10, std_MSE=5, alpha=0.2)
    # ps6_2_b("noisy_debate_hand", [15, 50, 140],
    #         num_particles=1000, std_control=10, std_MSE=2, alpha=0.4)
    ps6_3_a("pres_debate", [28, 84, 144], std_control=2, std_CHI=0.5)
