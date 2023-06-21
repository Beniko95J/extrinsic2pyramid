import numpy as np
from util.camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from load_llff import load_llff_data


def opengl_2_llff(poses):
    poses = np.concatenate([poses[:, :, 0:1], -poses[:, :, 1:2], -poses[:, :, 2:3], poses[:, :, 3:4]], axis=2)
    # poses = np.concatenate([poses[:, :, 0:1], poses[:, :, 1:2], poses[:, :, 2:3], poses[:, :, 3:4]], axis=2)

    return poses


datadir = '../data/stomach/subject03_04_100_200'
factor = 1
spherify = False
llffhold = 8

images, poses, bds, render_poses, i_test = load_llff_data(datadir, factor,
                                                          recenter=True, bd_factor=.75,
                                                          spherify=spherify)

i_test = np.arange(images.shape[0])[::llffhold]
i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                   (i not in i_test)])

train_poses = np.array(poses[i_train])
test_poses = np.array(poses[i_test])

# print(train_poses.shape)
# print(test_poses.shape)
# print(render_poses.shape)

train_poses = opengl_2_llff(train_poses)[:, :, 0:5]
test_poses = opengl_2_llff(test_poses)[:, :, 0:5]
render_poses = opengl_2_llff(render_poses)[:, :, 0:5]

bottom = np.array([[[0., 0., 0., 1.]]])
train_poses = np.concatenate([train_poses, np.repeat(bottom, len(train_poses), axis=0)], axis=1)
test_poses = np.concatenate([test_poses, np.repeat(bottom, len(test_poses), axis=0)], axis=1)
render_poses = np.concatenate([render_poses, np.repeat(bottom, len(render_poses), axis=0)], axis=1)

# Visualization
visualizer = CameraPoseVisualizer([-2, 2], [-2, 2], [-2, 2])
for pose in train_poses:
    visualizer.extrinsic2pyramid(pose, plt.cm.rainbow(0.1), 0.2)

for pose in test_poses:
    visualizer.extrinsic2pyramid(pose, plt.cm.rainbow(0.5), 0.2)

for pose in render_poses:
    visualizer.extrinsic2pyramid(pose, plt.cm.rainbow(0.9), 0.2)

# visualizer.customize_legend(['train', 'test', 'render_path'])
visualizer.show()
