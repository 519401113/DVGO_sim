import os
from os.path import join

# data_root = '/var/lib/docker/data/users/liwenye/Sim_NeRF/waymo_scenes'
# out_dir = '/var/lib/docker/data/users/liwenye/Sim_NeRF/Snerf/datasets'
data_root = '/SSD_DISK/users/chenyurui/PS/waymo_scenes'
out_dir = '/SSD_DISK/users/chenyurui/Snerf/mv_datasets'
os.makedirs(out_dir, exist_ok=True)
scenes = sorted(os.listdir(data_root))

for scene in scenes:
    depth_path = join(data_root,scene,'final_data')
    scene_path = join(data_root,scene)
    scene_name = join(out_dir, scene)
    if os.path.exists(join(scene_name, 'depth')):
        print('{} has depth'.format(scene))
        continue
    cmd = 'python waymo_preprocess.py --datadir {} --scene_name {} --depthdir {}'.\
        format(scene_path, scene_name, depth_path)

    os.system(cmd)

