
import numpy as np
from util.camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt
import os


def opengl_2_llff(poses):
    poses = np.concatenate([poses[:, :, 0:1], -poses[:, :, 1:2], -poses[:, :, 2:3], poses[:, :, 3:4]], axis=2)
    # poses = np.concatenate([poses[:, :, 0:1], poses[:, :, 1:2], poses[:, :, 2:3], poses[:, :, 3:4]], axis=2)

    return poses


datadir = '/home/zjiang/Documents/nerf_git/zipnerf_micware/exp/stomach/subject02_04_rgb/render'
train_poses = np.load(os.path.join(datadir, 'train_poses.npy'))
render_poses = np.load(os.path.join(datadir, 'render_poses.npy'))
# images, poses, bds, render_poses, i_test = load_llff_data(datadir, factor,
#                                                           recenter=True, bd_factor=.75,
#                                                           spherify=spherify)


# i_test = np.arange(images.shape[0])[::llffhold]
# i_train = np.array([i for i in np.arange(int(images.shape[0])) if
#                    (i not in i_test)])

# train_poses = np.array(poses[i_train])
# test_poses = np.array(poses[i_test])


# print(train_poses.shape)
# print(test_poses.shape)
# print(render_poses.shape)

train_poses = opengl_2_llff(train_poses)
render_poses = opengl_2_llff(render_poses)

bottom = np.array([[[0., 0., 0., 1.]]])
train_poses = np.concatenate([train_poses, np.repeat(bottom, len(train_poses), axis=0)], axis=1)
render_poses = np.concatenate([render_poses, np.repeat(bottom, len(render_poses), axis=0)], axis=1)

# # Visualization
visualizer = CameraPoseVisualizer([-2, 2], [-2, 2], [-2, 2])
for pose in train_poses:
    visualizer.extrinsic2pyramid(pose, plt.cm.rainbow(0.1), 0.1)

for pose in render_poses:
    visualizer.extrinsic2pyramid(pose, plt.cm.rainbow(0.9), 0.1)

# visualizer.customize_legend(['train', 'render_path'])
visualizer.show()
