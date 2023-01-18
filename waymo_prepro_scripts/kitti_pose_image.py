import numpy as np
import os 
import argparse

def convert_pose(args):
    lidar2cams = []
    K = []
    # CAMERA DIRECTION: RIGHT DOWN FORWARDS
    for i in sorted(os.listdir(os.path.join(args.datadir,'calib')),key=lambda x:int(x.split('.')[0])):
        f = open(os.path.join(args.datadir,'calib',i))
        raw = f.readlines()
        L = [raw[i].split()[1:] for i in range(len(raw))]
        K_ = np.array(L[:5]).reshape(-1,3,4)[:,:,:3]
        K.append(K_)
        lidar2cam = np.array(L[-5:])
        for j in range(len(lidar2cam)):
            I = np.eye(4)
            I[:3,:4] = lidar2cam[j].reshape(3,4)
            lidar2cams.append(I)
        f.close()

    K = np.stack(K,0).transpose([1,0,2,3])
    lidar2cams = np.stack(lidar2cams,0).reshape(-1,5,4,4).transpose([1,0,2,3])
    cam2lidar = np.linalg.inv(lidar2cams)
    return K, cam2lidar

def get_ego_pose(args):
    ego_poses=[]
    for i in sorted(os.listdir(os.path.join(args.datadir,'pose')),key=lambda x:int(x.split('.')[0])):
        ego_poses.append(np.loadtxt(os.path.join(args.datadir,'pose',i)))
    ego_poses = np.stack(ego_poses,0)
    return ego_poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str) # the file pathname  of data in kitti format
    parser.add_argument('--scene_name', type=str) # the file pathname  of data in Snerf format
    parser.add_argument('--pose_dir', type=str) # the pose directory
    
    args = parser.parse_args()
    os.makedirs(args.scene_name, exist_ok=True)

    # Decode poses
    ego_poses = get_ego_pose(args)
    K, cam2lidar = convert_pose(args)
    c2w = ego_poses @ cam2lidar

    # Save poses
    os.makedirs(args.scene_name, exist_ok=True)
    np.save(os.path.join(args.scene_name,'c2w.npy'), c2w)
    np.save(os.path.join(args.scene_name,'intrinsic.npy'), K)
