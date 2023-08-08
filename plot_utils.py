import numpy as np
from .util.camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt
import os


def opengl_2_llff(poses):
    poses = np.concatenate([poses[:, :, 0:1], -poses[:, :, 1:2], -poses[:, :, 2:3], poses[:, :, 3:4]], axis=2)
    # poses = np.concatenate([poses[:, :, 0:1], poses[:, :, 1:2], poses[:, :, 2:3], poses[:, :, 3:4]], axis=2)

    return poses


def plot_camera_poses_list(poses_list, bbox=2):
    visualizer = CameraPoseVisualizer([-bbox, bbox], [-bbox, bbox], [-bbox, bbox])

    color_list = np.linspace(0, 1, len(poses_list))
    for poses, color in zip(poses_list, color_list):
        poses = opengl_2_llff(poses)
        bottom = np.array([[[0., 0., 0., 1.]]])
        poses = np.concatenate([poses, np.repeat(bottom, len(poses), axis=0)], axis=1)
        for p in poses:
            visualizer.extrinsic2pyramid(p, plt.cm.rainbow(color), 0.1)

    visualizer.show()