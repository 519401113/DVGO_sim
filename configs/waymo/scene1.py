_base_ = '../nerf_unbounded/nerf_unbounded_default.py'

expname = 'dvgo_waymo'
basedir = './waymo/waymo_scenes'

data = dict(
    datadir='../Snerf/full_datasets/datasets/0032150',
    dataset_type='waymo',
    white_bkgd=False,
    datahold = 4,
    H=1280,
    W=1920,
)

# fine_model_and_render.update(dict(
#     num_voxels=160**3,
#     num_voxels_base=160**3,
#     rgbnet_dim=12,
#     alpha_init=1e-2,
#     fast_color_thres=1e-4,
#     maskout_near_cam_vox=False,
#     world_bound_scale=1.05,
# ))
