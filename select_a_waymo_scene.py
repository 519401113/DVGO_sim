import os
from os.path import join


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("Create dir: %s" % dir)

data_root = '/HDD_DISK/datasets/waymo/kitti_format/training'

tags = ['image_0','image_1','image_2','image_3','image_4','calib','velodyne','pose',
        'label_0','label_1','label_2','label_3','label_4','label_all']
posts = ['.png','.png','.png','.png','.png','.txt', '.bin','.txt'
         ,'.txt','.txt','.txt','.txt','.txt','.txt']

out_dir = './select_waymo_scenes'

name_list = sorted(os.listdir(join(data_root, tags[0])))
name_list = [name[:-4] for name in name_list]

scene_id = 17085
scene_range = 25

begin = scene_id-scene_range
end = scene_id+scene_range
# begin = 569130
# end = 569163


create_dir(out_dir)
scene_dir = join(out_dir, '%07d' % scene_id)
create_dir(scene_dir)

for tag in tags:
    create_dir(join(scene_dir, tag))

for name in name_list:
    if int(name) > end:
        break
    if int(name) < end and int(name) > begin:
        for post, tag in zip(posts,tags):
            cmd = "cp {} {}".format(join(data_root,tag,name+post), join(scene_dir, tag))
            os.system(cmd)






