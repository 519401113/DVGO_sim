import os, argparse
from os.path import join

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--gpu", type=int, default=0,
                    help='which gpu to use')
parser.add_argument("--num_part", type=int, default=1,
                    help='split the task to n part')
parser.add_argument("--ind", type=int, default=0,
                    help='which part, start from 0')

args = parser.parse_args()
n = args.num_part
ind = args.ind
gpu = args.gpu


all_config = sorted(os.listdir('configs/waymo_dataset'))
l = len(all_config)
bs = l//n
if ind<n-1:
    configs = all_config[ind*bs:ind*bs+bs]
else:
    configs = all_config[ind*bs:]

for config in configs:
    # if '0032150' in config:
    #     continue
    cmd = 'CUDA_VISIBLE_DEVICES={} python run.py --config ./configs/waymo_dataset/{} --render_test'.format(gpu,config)
    os.system(cmd)