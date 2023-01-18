# from fileinput import filename
from matplotlib.pyplot import axis
from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow as tf
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset.utils import camera_segmentation_utils
import numpy as np

print('success')
file_pathname='/SSD_DISK/users/zhangjunge/individual_files_training_segment-16608525782988721413_100_000_120_000_with_camera_labels.tfrecord'

file_data = tf.data.TFRecordDataset(file_pathname, compression_type='')
frames_with_seg = []
num_list=[]
sequence_id = None
for frame_num, frame_data in enumerate(file_data):
#   frame_num,frame_data = data
  frame = open_dataset.Frame()
  frame.ParseFromString(bytearray(frame_data.numpy()))
  # Save frames which contain CameraSegmentationLabel messages. We assume that
  # if the first image has segmentation labels, all images in this frame will.
  if frame.images[0].camera_segmentation_label.panoptic_label:
#     print('add!')

    frames_with_seg.append(frame)
    num_list.append(frame_num)
    if sequence_id is None:
      sequence_id = frame.images[0].camera_segmentation_label.sequence_id
    # Collect 3 frames for this demo. However, any number can be used in practice.
    if frame.images[0].camera_segmentation_label.sequence_id != sequence_id:
      break

# Organize the segmentation labels in order from left to right for viz later.
camera_left_to_right_order = [
                              open_dataset.CameraName.FRONT,
                              open_dataset.CameraName.FRONT_LEFT,
                              open_dataset.CameraName.FRONT_RIGHT,
                              open_dataset.CameraName.SIDE_LEFT,
                              open_dataset.CameraName.SIDE_RIGHT]
segmentation_protos_ordered = []
for frame in frames_with_seg:
  segmentation_proto_dict = {image.name : image.camera_segmentation_label for image in frame.images}
  segmentation_protos_ordered.append([segmentation_proto_dict[name] for name in camera_left_to_right_order])
# import pdb;pdb.set_trace()
segmentation_protos_flat = sum(segmentation_protos_ordered, [])
panoptic_labels, is_tracked_masks, panoptic_label_divisor = camera_segmentation_utils.decode_multi_frame_panoptic_labels_from_protos(
    segmentation_protos_flat, remap_values=True
)
NUM_CAMERA_FRAMES = 5
semantic_labels_multiframe = []
instance_labels_multiframe = []
for i in range(0, len(segmentation_protos_flat), NUM_CAMERA_FRAMES):
  semantic_labels = []
  instance_labels = []
  for j in range(NUM_CAMERA_FRAMES):
    semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
      panoptic_labels[i + j], panoptic_label_divisor)
    semantic_labels.append(semantic_label)
    instance_labels.append(instance_label)
  semantic_labels_multiframe.append(semantic_labels)
  instance_labels_multiframe.append(instance_labels)

def _pad_to_common_shape(label):
      return np.pad(label, [[0,1280 - label.shape[0]], [0, 0], [0, 0]])

semantic_labels = [[_pad_to_common_shape(label) for label in semantic_labels] for semantic_labels in semantic_labels_multiframe]
frame_nums = np.array(num_list)-1
Semantic = np.array(semantic_labels)
frame_nums = np.broadcast_to(frame_nums[:,None,None,None,None], Semantic.shape)
output = np.concatenate([frame_nums, Semantic],axis=-1)
import os
output_dir = './waymo_scene2'
# os.makedirs(os.path.join(output_dir, 'semantic_labels'),exist_ok=True)
np.save(os.path.join(output_dir,'semantic_labels.npy'),output)

import pdb;pdb.set_trace()