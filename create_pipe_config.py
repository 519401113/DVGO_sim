import os
from os.path import join
os.chdir('./configs/waymo')
cmd = 'python create_config.py'
os.system(cmd)