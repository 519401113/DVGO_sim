import os
# root_dir = '/var/lib/docker/data/users/liwenye/Snerf/'
root_dir = '/SSD_DISK/users/chenyurui/Snerf/'

dataset_dir = os.path.join(root_dir, 'full_datasets/datasets')
scene_list = sorted(os.listdir(dataset_dir))

# for idx, scene_name in enumerate(scene_list):
#     if '.' in scene_name:
#         continue
#     f = os.path.join('./','{}_{}.txt'.format(idx, scene_name))
#     with open(f, 'w') as file:
#         for arg in sorted(vars(args)):
#             attr = getattr(args, arg)
#             if type(attr)==str and '0032150' in attr:
#                 attr = attr.replace('0032150', scene_name)
#             if arg=='config':
#                 continue
#             file.write('{} = {}\n'.format(arg, attr))
base_config = './base.py'
import fileinput
for idx, scene_name in enumerate(scene_list):
    if '.' in scene_name:
        continue
    f = os.path.join('../waymo_dataset','{}_{}.py'.format(idx, scene_name))
    with open(f, 'w') as file:
        for line in fileinput.input(base_config):
            attr = line
            if '0032150' in line:
                attr = attr.replace('0032150', scene_name)
            file.write(attr)
