import math
import os
from os.path import join
import numpy as np
from math import cos, sin
import cv2


# vertices = 0.5*np.array([[1,1,1],[1,-1,-1],[-1,1,1],[-1,-1,-1],[1,1,1],[-1,1,-1],[1,-1,1],[-1,-1,-1],[1,1,1],[-1,-1,1],[1,1,-1],[-1,-1,-1]]).T
# the location actually point to the center of the bottom of an object
vertices = np.array([[0.5,1,0.5],[-0.5,-0,0.5],[0.5,1,-0.5],[-0.5,0,-0.5],[1,1,1],[-1,1,-1],[1,-1,1],[-1,-1,-1],[1,1,1],[-1,-1,1],[1,1,-1],[-1,-1,-1]]).T


v1 = np.array([[0.5,1,0.5],[0.5,1,-0.5],[-0.5,1,-0.5],[-0.5,1,0.5],]).T
v2 = np.array([[0.5,0,0.5],[0.5,0,-0.5],[-0.5,0,-0.5],[-0.5,0,0.5],]).T

v1 = np.array([[0.5,1,-0.5],[0.5,0,-0.5],[-0.5,0,-0.5],[-0.5,1,-0.5],]).T
v2 = np.array([[0.5,1,0.5],[0.5,0,0.5],[-0.5,0,0.5],[-0.5,1,0.5],]).T


class BBox():
    '''
    a single 3D bbox
    '''
    def __init__(self, cls, whl, location, rotation, score=1,
                 c2w=None, first_end=None, vector=None, label_path=None):
        self.cls = cls # the class of an object
        self.location = location # the center of an object's location
        self.whl = whl # 3D size of an object, y is height, xz is width and long
        self.score = score # the confidence of an bbox
        self.rotation = rotation # the rotation in the camera's axis
        self.first_end = first_end
        # self.axis = np.array([[cos(rotation),0, -sin(rotation)],[0,1,0],[sin(rotation), 0, cos(rotation)]]).T
        self.axis = np.array([[-sin(rotation),0, -cos(rotation)],[0,-1,0],[cos(rotation), 0, -sin(rotation)]]).T
        # a line of the object in the ground
        self.vector = self.axis[:,2:]

        self.weight = 1/(abs(location[2,0])+1)

        self.label_path = None
        if label_path:
            self.label_path = label_path.split('/')[-1]




        if c2w is not None: # change the location to the global world
            self.change_world(c2w)
        if vector is not None:
            self.vector = vector

    @classmethod
    def build_from_a_line(cls, line, c2w=None, label_path=None):
        # line is a line in kiiti format label.txt
        things = line.split(' ')

        # kitti format whl is saved as hwl
        clas, whl, location, rotation= things[0], \
                                       np.array([things[9],things[8],things[10]]).astype('float').reshape(3,1),\
                                       np.array([things[11],things[12],things[13]]).astype('float').reshape(3,1), \
                                       float(things[14])
        first_end = [(int(float(things[4])), int(float(things[5]))),
                     (int(float(things[6])), int(float(things[7])))]
        vector = None
        if len(things) > 17:
            vector = np.array([things[15],things[16],things[17]]).astype('float').reshape(3,1)
        return BBox(clas, whl, location, rotation, c2w=c2w, first_end=first_end, vector=vector, label_path=label_path)

    def change_world(self, c2w):
        ca = np.concatenate([self.location, np.ones([1,1])],axis=0)
        new_location = c2w @ ca
        self.location = new_location[:3]
        self.axis = c2w[:3,:3] @ self.axis
        self.vector =  c2w[:3,:3] @ self.vector
        # self.rotation

    def cal_rotation_from_vector(self):
        vector = self.vector
        self.rotation = math.atan2(-vector[2,0],vector[0,0])

    def cal_axis_from_rotation(self):
        rotation = self.rotation
        self.axis = np.array([[-sin(rotation), 0, -cos(rotation)], [0, -1, 0], [cos(rotation), 0, -sin(rotation)]]).T
    def change_all_to_camera_world(self, w2c):
        # only vector is belivable, we calculate the rotation and axis from vector
        self.change_world(w2c)
        self.cal_rotation_from_vector()
        self.cal_axis_from_rotation()


    @classmethod
    def cal_score(cls, bbox1, bbox2):
        '''
        :param bbox2:
        :return: a score about the similarity of 2 bboxes
        '''
        if bbox1.cls != bbox2.cls:
            return 0
        # if np.linalg.norm(bbox1.location-bbox2.location)>0.3*max(bbox1.whl.max(),bbox1.whl.max()):
        #     return 0
        # else:
        #     return 1
        score1 = np.linalg.norm(bbox1.location-bbox2.location)/max(bbox1.whl.max(),bbox2.whl.max())
        score2 = (bbox1.vector*bbox2.vector).sum()
        score3 = abs(((bbox1.location-bbox2.location)*bbox1.vector).sum())/\
                 (np.linalg.norm(bbox1.location-bbox2.location)+0.1)
        score4 = score1*(1.5-score3)
        if score1<0.1:
            return 1
        if score2<0.5 or score1>0.8:
            return 0
        return 1-2*score4

        # return 1-(((bbox1.location-bbox2.location)**2).sum())

    def visible(self):
        v11 = self.location + self.axis @ (self.whl * v1)
        v22 = self.location + self.axis @ (self.whl * v2)
        v = np.concatenate([v11, v22], -1)
        # if self.location[2,0] > 0:
        # import pdb; pdb.set_trace()
        if v[2].max() > 0:
            return True
        else:
            return False

    def draw_2d_box(self, img_path, K, save_path='test_2d.png'):
        image = cv2.imread(img_path)
        vv = np.concatenate([v1, v2], axis=1)
        vv = self.location + self.axis @ (self.whl*vv)
        vv = K @ vv
        vv = vv[:2]/vv[2]
        vv = vv.astype('int')

        first_point = vv.max(1)
        end_point = vv.min(1)
        cv2.rectangle(image, first_point, end_point, (0, 255, 0), 2)

        cv2.imwrite(save_path, image)

    def draw_3d_box(self, img_path, K, save_path='test_3d.png', if_return=False):

        v11 = self.location + self.axis @ (self.whl*v1)
        v11 = K @ v11
        v11 = v11[:2]/abs(v11[2])
        v11 = v11.astype('int')
        v22 = self.location + self.axis @ (self.whl*v2)
        v22 = K @ v22
        v22 = v22[:2]/abs(v22[2])
        v22 = v22.astype('int')
        if if_return:
            return v11,v22
        image = cv2.imread(img_path)
        for i in range(4):
            cv2.line(image, v11[:,i], v11[:,((i+1)%4)], (0, 255, 0), 2)
            cv2.line(image, v22[:, i], v22[:, ((i + 1) % 4)], (0, 0, 255), 2)
            cv2.line(image, v22[:, i], v11[:, i], (255, 0, 0), 2)

        cv2.imwrite(save_path, image)



import copy
class Object_List():
    '''
    a list of objects, each objects have a list of candidate bbox
    '''
    def __init__(self, bboxs, global_world=False):
        self.bbox_list = bboxs # list of bboxs: [[bbox11, bbox12, ...], ...]
        self.bbox_attr = [] # after fuse the bboxes, the std of angel and position
        if bboxs is None:
            self.bbox_list = []
        self.global_bbox = None
        if global_world:
            self.global_bbox = copy.deepcopy(bboxs)

    def resume_from_global(self):
        self.bbox_list = copy.deepcopy(self.global_bbox)

    def __len__(self):
        return len(self.bbox_list)

    def __getitem__(self, item):
        return self.bbox_list[item]

    def fuse_self(self):
        for i in range(len(self.bbox_list)):
            object = self.bbox_list[i]
            if len(object) > 1:
                # weighted fuse
                weight = sum([a.weight for a in object])
                mean_location = np.stack([a.location*a.weight for a in object]).sum(0)/weight
                mean_vector = np.stack([a.vector*a.weight for a in object]).sum(0)/weight
                mean_whl = np.stack([a.whl*a.weight for a in object]).sum(0)/weight
                mean_rotation = np.array([a.rotation*a.weight for a in object]).sum()/weight
                mean_cls = object[0].cls
                bbox = BBox(cls=mean_cls, location=mean_location, whl=mean_whl, rotation=mean_rotation,
                            vector=mean_vector)
                self.bbox_list[i] = [bbox]

                std = np.linalg.norm(np.stack([a.location for a in object]).std(0))
                thr = 0.2
                if std>thr:
                    moving = True
                else:
                    moving = False

                attr = {"std_location": np.stack([a.location for a in object]).std(0),
                       "std_whl": np.stack([a.whl for a in object]).std(0),
                        "std_rotation": np.array([a.rotation for a in object]).std(),
                        "moving": moving,
                        "bbox_dict": {a.label_path: a for a in object},}
                        # "label_ids": [a.label_path for a in object],
                        # "bbox_list": copy.deepcopy(object)}
                self.bbox_attr.append(attr)

            # if len(object) > 1:
            #     # the simplest way of mean fuse
            #     mean_location = np.stack([a.location for a in object]).mean(0)
            #     mean_vector = np.stack([a.vector for a in object]).mean(0)
            #     mean_whl = np.stack([a.whl for a in object]).mean(0)
            #     mean_rotation = np.array([a.rotation for a in object]).mean()
            #     mean_cls = object[0].cls
            #     bbox = BBox(cls=mean_cls, location=mean_location, whl=mean_whl, rotation=mean_rotation,
            #                 vector=mean_vector)
            #     self.bbox_list[i] = [bbox]

    def remove_moving(self):
        static_bbox_list = []
        static_bbox_attr = []
        for i, attr in enumerate(self.bbox_attr):
            if not attr['moving']:
                static_bbox_list.append(self.bbox_list[i])
                static_bbox_attr.append(attr)
        self.global_bbox = copy.deepcopy(self.bbox_list)
        self.global_attr = copy.deepcopy(self.bbox_attr)
        self.bbox_list = static_bbox_list
        self.bbox_attr = static_bbox_attr


    def change_world(self, c2w):
        for l in self.bbox_list:
            for j in l:
                j.change_all_to_camera_world(c2w)


    def draw_3d_box(self, img_path, K, save_path='./save3d.png'):
        save = 0
        for ind, i in enumerate(self.bbox_list):
            bbox = i[-1]
            if save == 0:
                if bbox.visible():
                    bbox.draw_3d_box(img_path, K, save_path)
                    save = 1
            else:
                if bbox.visible():
                    bbox.draw_3d_box(save_path, K, save_path)
        return save

    def create_mask(self, intrinsic, path=None, if_pad=False, debug=False):
        # if debug:
        #     import pdb; pdb.set_trace()
        mask_all = np.zeros([1280,1920])
        cat = {'Car':2, 'Cyclist': 3, 'Pedestrian': 7}
        sorted_bbox = self.bbox_list.sort(key=lambda x: -x[-1].location[2,0])
        sorted_bbox = self.bbox_list
        for ind, i in enumerate(sorted_bbox):
            bbox = i[-1]
            s = cat[bbox.cls]
            if bbox.visible():
                mask = np.zeros([1280, 1920])
                v11, v22 = bbox.draw_3d_box(img_path=None, K=intrinsic, if_return=True)
                v = np.concatenate([v11,v22], -1)
                max_v = v.max(-1)
                min_v = v.min(-1)

                # # 去除异常的mask
                # if (max_v-min_v).max() > 2000:
                #     continue
                cds = [[max_v.tolist(),[max_v[0], min_v[1]], min_v.tolist(), [min_v[0], max_v[1]]]]
                cds = np.array(cds)
                cv2.fillPoly(mask, cds, 1)
                mask_all = mask_all*(1-mask) + mask*s*25
        if if_pad:
            mask_all[886:] = 255

        # import pdb; pdb.set_trace()
        if path:
            cv2.imwrite(path, mask_all)







    @classmethod
    def fuse_2_list(cls, objlist1, objlist2):
        # 最大分数匹配
        for i in range(len(objlist2)):
            match = 0
            o2 = objlist2[i]
            # scores = []
            max_match = 0
            max_score = 0
            for j in range(len(objlist1)):
                o1 = objlist1[j]
                score = BBox.cal_score(o1[-1], o2[-1])
                if score > max_score:
                    max_score = score
                    max_match = j
            if max_score>0:
                objlist1.bbox_list[max_match] += o2
            else:
                objlist1.bbox_list.append(objlist2[i])

            # for j in range(len(objlist1)):
            #     o1 = objlist1[j]
            #     score = BBox.cal_score(o1[-1], o2[-1])
            #     if score > 0.9:
            #         match = 1
            #         objlist1.bbox_list[j] += o2
            #         break
            # if match == 0:
            #     objlist1.bbox_list.append(objlist2[i])
        return objlist1

    @classmethod
    def build_from_a_path(cls, label_path=None, c2w=None, visible=False, global_world=False):
        f = open(label_path)
        raw = f.readlines()
        raw = sorted(list(set(line for line in raw)))
        if visible:
            # reserve the visible bboxes
            raw = [line for line in raw if float(line.split()[13]) > 0]

        return Object_List([[BBox.build_from_a_line(line=line, c2w=c2w, label_path=label_path)] for line in raw], global_world=global_world)

    def save(self, save_path='test.txt', global_world=True, cover=True):
        '''
        :param save_path:
        :param global_world: if the bboxes are saved in the global world, the vector should be saved
                             to identify the rotation, and the rotation (0) saved is wrong
        :param cover: if cover the raw data, or renewal
        :return:
        '''
        mod = 'w' if cover else 'a'
        with open(save_path, mod) as file:
            for l in self.bbox_list:
                bbox = l[-1]
                if global_world:
                    line = '{} 0 0 -10 0 0 0 0 {} {} {} {} {} {} 0 {} {} {}\n'.format(bbox.cls, bbox.whl[1, 0],
                                                                                      bbox.whl[0, 0], bbox.whl[2, 0],
                                                                                      bbox.location[0, 0],
                                                                                      bbox.location[1, 0],
                                                                                      bbox.location[2, 0],
                                                                                      bbox.vector[0, 0],
                                                                                      bbox.vector[1, 0],
                                                                                      bbox.vector[2, 0])
                    file.write(line)
                else:
                    if bbox.visible():
                        bbox.cal_rotation_from_vector()
                        line = '{} 0 0 -10 0 0 0 0 {} {} {} {} {} {} {}\n'.format(bbox.cls, bbox.whl[1, 0],
                                                                                  bbox.whl[0, 0], bbox.whl[2, 0],
                                                                                  bbox.location[0, 0],
                                                                                  bbox.location[1, 0],
                                                                                  bbox.location[2, 0],
                                                                                  bbox.rotation)
                        file.write(line)



def create_mask(scene_dir, dataset_dir, num=50):
    '''
    :param scene_dir: raw data dir, which has image_x, label_x, ...
    :param dataset_dir: processed data dir, which has c2w.npy, intrinsic.npy, image, depth, ...
    :param num: processed image num
    :return:
    '''
    c2ws = np.load(join(dataset_dir, 'c2w.npy'))
    intrinsics = np.load(join(dataset_dir, 'intrinsic.npy')).astype('float')
    HW = (1280, 1920)
    cam_id = 0
    start = 0
    frames = sorted(os.listdir(join(scene_dir, 'image_0')))
    num = min(len(frames),1000)
    label_root = scene_dir

    for i,frame in enumerate(frames):
        # label_path = join(raw_data, scene, 'label_{}'.format(cam_id), frame.replace('.png', '.txt'))
        # import pdb; pdb.set_trace()
        label_path = join(label_root, 'label_all', frame.replace('.png', '.txt'))

        if not os.path.exists(label_path):
            continue
        if start == 0:
            object1 = Object_List.build_from_a_path(label_path, c2ws[cam_id, i], visible=False)
            start = 1
        else:
            object2 = Object_List.build_from_a_path(label_path, c2ws[cam_id, i], visible=False)
            object1 = Object_List.fuse_2_list(object1, object2)

    if start == 0:
        print('|------ Nothing to add ------|')
        return 0
    object1.fuse_self()
    print('|------ fuse bbox lists complete ------|')

    os.makedirs(join(dataset_dir, 'mask'), exist_ok=True)
    movings = []
    for attr in object1.bbox_attr:
        if attr['moving']:
            movings.append(attr['bbox_dict'])
    if len(movings) == 0:
        pass

    for i, frame in enumerate(frames):
        if i>=num:
            break
        label = frame.replace('png','txt')
        i_bbox = []
        for obj in movings:
            if label in obj:
                i_bbox.append(obj[label])
        if len(i_bbox) == 0:
            continue



        mvoblist = Object_List([[a] for a in i_bbox], global_world=True)
        print('|------ save masks of {} ------|'.format(frame))
        for j in range(5):
            mask_id = num*j+i
            c2w = c2ws[j, i]
            intrinsic = intrinsics[j, i]
            mvoblist.change_world(np.linalg.inv(c2w))

            # if i==9:
            #     tt = Object_List([[mvoblist[2][0]]])
            #     tt.create_mask(intrinsic=intrinsic, path='test.png', debug=True)

            if_pad = False if j < 3 else True
            mvoblist.create_mask(intrinsic=intrinsic, path=join(dataset_dir, 'mask', "%04d.png"%mask_id), if_pad=if_pad)
            mvoblist.resume_from_global()








def add_bbox(scene_path, result_path, dataset_path):
    c2w = np.load(join(dataset_path, 'c2w.npy'))
    frames = sorted(os.listdir(join(scene_path, 'image_0')))
    t = 0


    for idx,frame in enumerate(frames):
        label_path = join(scene_path, 'label_all', frame.replace('png','txt'))
        if not os.path.exists(label_path):
            continue
        if idx>49:
            break
        # import pdb; pdb.set_trace()
        if t == 0:
            object1 = Object_List.build_from_a_path(label_path, c2w[0, idx])
            t = 1
        else:
            object2 = Object_List.build_from_a_path(label_path, c2w[0, idx])
            object1 = Object_List.fuse_2_list(object1, object2)

    if t == 0:
        print('Nothing to add')
        return 0
    obj_path = join(dataset_path, 'all_obj.txt')
    object1.fuse_self()
    object1.remove_moving()
    object1.save(obj_path)

    # import pdb; pdb.set_trace()
    all_obj = Object_List.build_from_a_path(obj_path, global_world=True,visible=False)
    c2w = np.load(join(result_path, 'target_poses.npy'))
    intrinsic = np.load(join(result_path, 'intrinsic.npy'))
    label_paths = sorted(os.listdir(join(result_path,'bbox')))
    assert c2w.shape[0] == len(label_paths), 'Invaild c2w and labels for {}'.format(result_path)

    os.makedirs(join(result_path,'vis'),exist_ok=True)
    for ind, label_path in enumerate(label_paths):
        all_obj.resume_from_global()
        all_obj.change_world(np.linalg.inv(c2w[ind]))
        all_obj.save(join(result_path,'bbox',label_path), global_world=False, cover=False)

        id_bboxs = Object_List.build_from_a_path(join(result_path,'bbox',label_path), global_world=False,visible=False)
        id_bboxs.save(join(result_path,'bbox',label_path), global_world=False, cover=True)
        if ind < 5:
            id_bboxs.draw_3d_box(join(result_path, 'image', label_path.replace('txt', 'png')), intrinsic,
                                join(result_path, 'vis', label_path.replace('txt', 'png')))


    print('Add bbox for {}'.format(result_path))
    return 1


def pipe_add():
    data_root = 'annotation'
    mod = os.listdir(data_root)
    mod = [m for m in mod if 'dvgo_' in m]
    for m in mod:
        scene_id = m[5:]
        scene_path = join('waymo_scenes', scene_id)
        dataset_path = join('Snerf','full_datasets','datasets',scene_id)
        results = os.listdir(join('annotation',m))
        for result in results:
            result_path = join('annotation',m,result)
            things = os.listdir(result_path)
            if 'bbox' in things and 'target_poses.npy' in things:
                add_bbox(scene_path, result_path, dataset_path)

    # scene_path = 'waymo_scenes/0032150'
    # result_path = 'annotation/dvgo_0032150/1671098605'
    # dataset_path = 'Snerf/full_datasets/datasets/0032150'
    # aa = add_bbox(scene_path=scene_path, result_path=result_path, dataset_path=dataset_path)
    import pdb; pdb.set_trace()

if __name__ == '__main__':

    data_root = '/SSD_DISK/users/chenyurui/PS/waymo_scenes' ## waymo kitti format scenes
    out_dir = '/SSD_DISK/users/chenyurui/Snerf/mv_datasets' ## DVGO dataloader format dataset dir
    os.makedirs(out_dir, exist_ok=True)
    scenes = sorted(os.listdir(data_root))
    # import pdb; pdb.set_trace()

    # scene_index = '0047140'
    # dataset_dir = join(out_dir, scene_index)
    # scene_dir = join(data_root, scene_index)
    # create_mask(scene_dir, dataset_dir)

    for scene_index in scenes:
        dataset_dir = join(out_dir, scene_index)
        scene_dir = join(data_root, scene_index)
        create_mask(scene_dir, dataset_dir)

