# DVGO for sim
### train
python run.py --config configs/waymo/base.py

config中data.datadir指向训练场景地址
### 挑选场景
python select_a_scene.py --scene_id xxxx   （xxxx为数字或完整序号，如0032150或32150）

选择waymo kitti format中序号为xxxx的前24帧和后24帧（共49帧）的单场景数据集，该文件中的data_root为waymo数据集存放点，
out_dir为挑选出场景的存放点，假设为waymo_scenes。
生成完毕后会有目录waymo_scenes/xxxx。

可以挑选动态物体较少的场景，或者感觉这些动态物体被mask后不怎么影响背景的训练（一般来说大部分的有动态物体的场景都能训）

### 处理场景以作为DVGO的训练集
1. 将out_dir/xxxx交由lwy生成depth map
2. cd waymo_prepro_scripts & python process_scenes.py，该文件中data_root指向waymo_scenes，out_dir为处理后的训练集生成点，设为datasets，则会生成datasets/xxxx。该脚本批量处理场景
3. 生成移动物体mask，python debug_bbox --scene_index xxxx(7位数，如0032150) --dataset_dir datasets --scene_dir waymo_scenes，改脚本处理单一场景xxxx。
4. datasets/xxxx即可用来训练DVGO
