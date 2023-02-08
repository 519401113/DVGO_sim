_base_ = '../nerf_unbounded/nerf_unbounded_default.py'

expname = "0087070"
basedir = "./waymo_ckpt/pipe_train"

data = dict(
    datadir='../Snerf/mv_datasets/0087070',
    dataset_type='waymo',
    white_bkgd=False,
    datahold = 4,
    H=1280,
    W=1920,
)

alpha_init = 1e-4
stepsize = 0.3

# coarse_train = (dict(
#     depth_loss=0.01
# ))

fine_train = dict(
    N_iters=40000,
    N_rand=4096,
    lrate_s0=1e-1,  # lr of color/feature voxel grid
    lrate_segnet=1e-3,  # lr of the mlp to preduct view-dependent color
    lrate_decay=80,
    ray_sampler='flatten',
    weight_nearclip=1.0,
    weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000,10000,12000,14000,16000],
    tv_before=20000,
    tv_dense_before=20000,
    weight_tv_density=1e-6,
    weight_tv_k0=1e-7,
    patch_size=[32,32],

    posenet_config=dict(
        if_refine=False,  ## true for 4h, false for 1h
        learn_r=True,
        learn_t=False,
        lr=1e-5,
        begin=1e4,
        end=2e4
    )

)
fine_train.update(dict(
    N_iters=60000,
    depth_loss = 1,
    smoothness_loss = 0.01,
    # ray_sampler='flatten',
    ray_sampler='mixed',
    bounding_near=False,
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

    # segmetation params
    seg_config = dict(
        segnet_dim=12,  # ori 12
        segnet_depth=3,
        segnet_width=512,
        s0_type='DenseGrid',
        s0_config=dict(),
        seg_class=19,
        # let rgb and seg use the same feature grid, the k0 feature dim would be seg+rgb
        fuse_rgb_and_seg=False,
    ),
)

fine_model_and_render.update(dict(
    num_voxels=320**3,   # ori 320, 420
    num_voxels_base=320**3, # ori 320
    rgbnet_dim=24,                # ori 12, v1 24
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=512,             # width of the colors MLP
    alpha_init=1e-2,
    fast_color_thres=1e-4,
    maskout_near_cam_vox=False,
    world_bound_scale=1.05,
))
