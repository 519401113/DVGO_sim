_base_ = '../nerf_unbounded/nerf_unbounded_default.py'

expname = 'dvgo_waymo_v1'
basedir = './waymo/waymo_scenes'

data = dict(
    datadir='../Snerf/full_datasets/datasets/0032150',
    dataset_type='waymo',
    white_bkgd=False,
    datahold = 4,
    H=1280,
    W=1920,
)

alpha_init = 1e-4
stepsize = 0.3

fine_train = dict(
    N_iters=40000,
    N_rand=4096,
    lrate_decay=80,
    ray_sampler='flatten',
    weight_nearclip=1.0,
    weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000,10000,12000,14000,16000],
    tv_before=20000,
    tv_dense_before=20000,
    weight_tv_density=1e-6,
    weight_tv_k0=1e-7,
)
fine_train.update(dict(
    N_iters=60000,
    depth_loss = 0
))
fine_model_and_render = dict(
    num_voxels=320**3,
    num_voxels_base=320**3,
    alpha_init=alpha_init,
    stepsize=stepsize,
    fast_color_thres={
        '_delete_': True,
        0   : alpha_init*stepsize/10,
        1500: min(alpha_init, 1e-4)*stepsize/5,
        2500: min(alpha_init, 1e-4)*stepsize/2,
        3500: min(alpha_init, 1e-4)*stepsize/1.5,
        4500: min(alpha_init, 1e-4)*stepsize,
        5500: min(alpha_init, 1e-4),
        6500: 1e-4,
    },
    world_bound_scale=1,
)

fine_model_and_render.update(dict(
    rgbnet_dim=12,
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=512,             # width of the colors MLP
    alpha_init=1e-2,
    fast_color_thres=1e-4,
    maskout_near_cam_vox=False,
    world_bound_scale=1.05,
))
