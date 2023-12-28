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
    # NOTE(wjh) reverse the mask!
    mask = np.where(mask > 0, 0.0, 1.0).astype(np.float32)

    return np.expand_dims(mask, axis=-1)


class MyDataset(Dataset):
    def __init__(self, root_path='datasets/sun360_d1_t30000_v03000', mask_path=None, num_training=500,
                 exclude_360=False):
        self.root_path = root_path
        self.num_training = num_training
        # NOTE(wjh) change to train_no360.json, exclude the panoramic part of the image
        if exclude_360:
            train_json_path = os.path.join(self.root_path, 'train_no360.json')
        else:
            train_json_path = os.path.join(self.root_path, 'train.json')  # train.json
        self.data = json.load(open(train_json_path, 'r'))[:self.num_training]
        self.height, self.width = 256, 512
        self.mask = read_mask(mask_path, self.height, self.width)
        # TODO(wjh)
        # warp the image into the mask, instead of directly using the mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.root_path, source_filename))
        target = cv2.imread(os.path.join(self.root_path, target_filename))


        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        # NOTE(wjh): only output rgb layers. Do we need to address the None part as 0?
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, mask=self.mask)