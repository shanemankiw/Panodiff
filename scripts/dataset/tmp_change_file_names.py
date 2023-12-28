import os
from tqdm import tqdm

folder = '/HDD/22Ubuntu/nfs_share/sun360/full'

paths = sorted(os.listdir(folder))

for path in tqdm(paths):
    if '_' in os.path.basename(path):

        file_name = os.path.basename(path).split('_')[-1] 
        os.system('mv {} {}'.format(os.path.join(folder, path), os.path.join(folder, file_name)))