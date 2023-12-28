import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


def read_mask(path, height, width):
    '''
    mask
    '''
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height))
    mask = np.where(mask>0, 1.0, 0).astype(np.float32)

    return np.expand_dims(mask, axis=-1)


class MyDataset(Dataset):
    def __init__(self, root_path='datasets/sun360_d1_t30000_v03000'):
        self.root_path = root_path
        
        self.height, self.width = 180, 320
        self.mask = read_mask(os.path.join(self.root_path, 'mask.jpg'), self.height*2, self.width*2)
        #target_filled = np.zeros([self.height*2, self.width*2, 3])
        #target_filled[self.mask[...,0]>0] = cv2.resize(cv2.imread(os.path.join(self.root_path, 'nov.jpg')), (self.width, self.height)).reshape([-1,3])

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        source_filename = 'nov.jpg'
        target_filename = source_filename
        prompt = 'road with side tree'

        #source = cv2.imread(os.path.join(self.root_path, source_filename))
        target = cv2.imread(os.path.join(self.root_path, target_filename))
        target = cv2.resize(target, (self.width, self.height))
        

        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target_filled = np.zeros([self.height*2, self.width*2, 3])
        target_filled[self.mask[...,0]>0] = target.reshape([-1,3])
        source = target_filled.copy()

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        # NOTE(wjh): only output rgb layers. Do we need to address the None part as 0?
        target_filled = (target_filled.astype(np.float32) / 127.5) - 1.0

        # ugly fix
        target_filled = np.pad(target_filled, ((12, 12), (0, 0), (0, 0)), mode='constant', constant_values=-1)
        source = np.pad(source, ((12, 12), (0, 0), (0, 0)), mode='constant', constant_values=0)
        mask = np.pad(1-self.mask, ((12, 12), (0, 0), (0, 0)), mode='constant', constant_values=0)

        return dict(jpg=target_filled, txt=prompt, hint=source, mask=mask)

