import numpy as np
from PIL import Image
from matplotlib import cm
import os
import matplotlib.pyplot as plt


def visualize_depth(depth, pathname, idx):
    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = depth.min() + eps
    far = depth.max() - eps
    near, far, depth = [curve_fn(x) for x in [near, far, depth]]
    depth = np.nan_to_num(
        np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(depth)[:, :, :3]
    os.makedirs('./test_depths/' + pathname, exist_ok=True)
    Image.fromarray(
        (np.clip(np.nan_to_num(vis), 0., 1.) * 255.).astype(np.uint8)).save(
        os.path.join('./test_depths/', pathname, 'depth_vis_{:04d}.png'.format(idx)))


def visualize_semantic(semantic, pathname, idx):
    plt.figure(figsize=(64, 60))
    plt.imshow(semantic)
    plt.grid(False)
    plt.axis('off')
    os.makedirs(os.path.join(pathname, 'test_semantics'), exist_ok=True)
    plt.savefig(os.path.join(pathname, 'test_semantics', 'semantic_vis_{:04d}.png'.format(idx)))
    plt.clf()