import json
from PIL import Image
import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from RotationModels.utils.compute_utils import inverse_normalize
import cv2

class RotationDataset(VisionDataset):
    def __init__(self, root, pairs_file=None, num_images=1000, extensions='.jpg', height=256, Train=True, down_scale=1):
        self.height = height
        self.num_images = num_images
        assert down_scale == 1 or down_scale == 2, 'only support resolution of 1024X512 and 512X256'

        self.downscale = down_scale
        transform, target_transform = self.init_crops_transform()
        
        super(RotationDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        
        self.pairs = np.load(pairs_file, allow_pickle=True).item()
        self.extensions = extensions
        self.train = Train
        prompt_json_file_train = './prompts/my_sun360_prompts_no360.json'
        self.prompts = json.load(open(prompt_json_file_train, 'r'))

    def __getitem__(self, index):
        img1 = self.pairs[index]['img1']
        img2 = self.pairs[index]['img2']
        pano = self.pairs[index]['pano']
        
        path = os.path.join(self.root, img1['path'])
        rotation_x1, rotation_y1 = img1['x'], img1['y']
        image1 = self.loader(path)
        
        path2 = os.path.join(self.root, img2['path'])
        rotation_x2, rotation_y2 = img2['x'], img2['y']
        image2 = self.loader(path2)
        
        full_path = os.path.join(self.root, pano['path'])
        
        # NOTE: After this transformation, 
        # the range of the RGB channels is not [0,1] as in Stable Diffusion. 
        # This means image1 and image2 is only used for the RotaNet. 
        # If you want to generate inputs of ControlNet by warpping image1 and image2
        # please re-transfrom them
        image1_original = self.normal_transform(image1)
        image2_original = self.normal_transform(image2)
        if self.target_transform is not None:
            image1 = self.target_transform(image1)
            image2 = self.target_transform(image2)
        
        # RGB channel range=[0., 1.] dtype=torch.float32
        image_pano = self.normal_transform(self.loader(full_path, self.downscale))
        
        img_idx = full_path.split('/')[-2]
        prompt = self.prompts[img_idx]

        return {
            'img1': image1,
            'img2': image2,
            'img1_original': image1_original,
            'img2_original': image2_original,
            'pano': image_pano,
            'rotation_x1': rotation_x1,
            'rotation_y1': rotation_y1,
            'rotation_x2': rotation_x2,
            'rotation_y2': rotation_y2,
            'path': path,
            'path2': path2,
            'pano_path': full_path,
            'txt':prompt,
        }

    def __len__(self):
        # NOTE: only 1000 for test
        '''if len(self.pairs) > 1000 and not self.train:
            return 1000'''
        if len(self.pairs) > self.num_images:
            return self.num_images
        
        return len(self.pairs)
        
    def init_crops_transform(self):
        transform = transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        target_transform=transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        
        return transform, target_transform
    
    
    def normal_transform(self, image):
        image_arr = np.array(image)
        image_arr = (image_arr / 255).astype(np.float32)
        return torch.tensor(image_arr)
    
    def loader(self, path, down_scale=1):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if down_scale > 1:
            width, height = img.size
            new_width = int(width / down_scale)
            new_height = int(height / down_scale)
            new_size = (new_width, new_height)
            img = img.resize(new_size)
        return img


def read_mask(path, height, width):
    '''
    mask
    '''
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height))
    mask = np.where(mask>0, 1.0, 0).astype(np.float32)

    return np.expand_dims(mask, axis=-1)

# depracated
class SingleImageDataset(VisionDataset):
    def __init__(self, root, pairs_file=None, num_images=1000, extensions='.jpg', height=256, Train=True, down_scale=1):
        self.height = 512
        self.width = 1024
        self.num_images = num_images
        assert down_scale == 1 or down_scale == 2, 'only support resolution of 1024X512 and 512X256'

        self.downscale = down_scale
        transform, target_transform = self.init_crops_transform()
        
        super(SingleImageDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        
        self.pano_frames = sorted(os.listdir(root))
        self.extensions = extensions
        self.train = Train
        prompt_json_file_train = './prompts/my_sun360_prompts_no360.json'
        self.prompts = json.load(open(prompt_json_file_train, 'r'))
        self.mask = read_mask('datasets/90binarymask.png', self.height, self.width)

    def __getitem__(self, index):
        pano = self.pano_frames[index]
        
        full_path = os.path.join(self.root, pano, 'panorama.jpg')
        target = cv2.imread(full_path)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        source = target.copy()
        source[(self.mask==0)[..., 0]] = 0
        
        # RGB channel range=[0., 1.] dtype=torch.float32
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        img_idx = full_path.split('/')[-2]
        prompt = self.prompts[img_idx]

        return dict(jpg=target, txt=prompt, hint=source, mask=np.where(self.mask==0, 1, 0))

    def __len__(self):
        # NOTE: only 1000 for test
        '''if len(self.pairs) > 1000 and not self.train:
            return 1000'''
        return len(self.pano_frames)
            
    def init_crops_transform(self):
        transform = transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        target_transform=transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        
        return transform, target_transform
    
    
    def normal_transform(self, image):
        image_arr = np.array(image)
        image_arr = (image_arr / 255).astype(np.float32)
        return torch.tensor(image_arr)
    
    def loader(self, path, down_scale=1):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if down_scale > 1:
            width, height = img.size
            new_width = int(width / down_scale)
            new_height = int(height / down_scale)
            new_size = (new_width, new_height)
            img = img.resize(new_size)
        return img

class SingleDataset(VisionDataset):
    def __init__(self, root, dataset_name, index,
                prompt,
                prompt_path=None, extensions='.jpg', 
                height=512, Train=True, down_scale=1, 
                break_iter=None):
        
        # self.num_images = num_images
        assert down_scale == 1 or down_scale == 2, 'only support resolution of 1024X512 and 512X256'

        self.downscale = down_scale
        if self.downscale == 2:
            self.height = 256
            self.width = 512
        elif self.downscale == 1:
            self.height = 512
            self.width = 1024
        transform, target_transform = self.init_crops_transform()
        
        super(SingleDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        
        if break_iter is None:
            self.break_iter = 0
        else:
            self.break_iter = break_iter + 1
        self.extensions = extensions
        self.train = Train
        self.dataset_type = dataset_name
        prompt_json_file_train = prompt_path
        self.prompts = json.load(open(prompt_json_file_train, 'r'))

        # TODO: add these paths when ziyu told me to.
        mask_path = 'datasets/90binarymask.png'
        target_path = os.path.join(self.root, '{}/panorama.jpg'.format(index))

        self.mask = read_mask(mask_path, height=self.height, width=self.width)
        
        # reading the full pano
        target = cv2.imread(target_path)
        
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        source = target * self.mask

        self.target = (target.astype(np.float32) / 127.5) - 1.0

        # get hint
        #source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        self.hint = source.astype(np.float32) / 255.0
        
        self.prompt = prompt

    def __getitem__(self, index):
        
        return dict(jpg=self.target, 
                    txt=self.prompt, 
                    hint=self.hint, 
                    mask=np.where(self.mask==0, 1, 0))

    def __len__(self):
        
        return 5 # generate 5 instances for the same prompt
        
    def init_crops_transform(self):
        transform = transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        target_transform=transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        
        return transform, target_transform
    
    
    def normal_transform(self, image):
        image_arr = np.array(image)
        image_arr = (image_arr / 255).astype(np.float32)
        return torch.tensor(image_arr)
    
    def loader(self, path, down_scale=1):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if down_scale > 1:
            width, height = img.size
            new_width = int(width / down_scale)
            new_height = int(height / down_scale)
            new_size = (new_width, new_height)
            img = img.resize(new_size)
        return img

class PairDataset(VisionDataset):
    def __init__(self, root, dataset_name, index,
                prompt,
                pairs_file=None, extensions='.jpg', 
                height=512, Train=True, down_scale=1, 
                break_iter=None):
        # self.num_images = num_images
        assert down_scale == 1 or down_scale == 2, 'only support resolution of 1024X512 and 512X256'

        self.downscale = down_scale
        if self.downscale == 2:
            self.height = 256
            self.width = 512
        elif self.downscale == 1:
            self.height = 512
            self.width = 1024
        transform, target_transform = self.init_crops_transform()

        self.pairs = np.load(pairs_file, allow_pickle=True).item()

        # HACK: locate the index by this.
        num_panos = len(os.listdir(root)) - 1 # minus a meta folder
        num_pairs = len(self.pairs) // num_panos
        self.index = num_pairs * int(index) + 1
        
        super(PairDataset, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        
        if break_iter is None:
            self.break_iter = 0
        else:
            self.break_iter = break_iter + 1
        self.extensions = extensions
        self.train = Train
        self.dataset_type = dataset_name
                
        self.prompt = prompt

    def __getitem__(self, index):
        img1 = self.pairs[self.index]['img1'] # fixed index from input
        img2 = self.pairs[self.index]['img2']
        pano = self.pairs[self.index]['pano']
        
        path = os.path.join(self.root, img1['path'])
        rotation_x1, rotation_y1 = img1['x'], img1['y']
        image1 = self.loader(path)
        
        path2 = os.path.join(self.root, img2['path'])
        rotation_x2, rotation_y2 = img2['x'], img2['y']
        image2 = self.loader(path2)
        
        full_path = os.path.join(self.root, pano['path'])
        image1_original = self.normal_transform(image1)
        image2_original = self.normal_transform(image2)
        if self.target_transform is not None:
            image1 = self.target_transform(image1)
            image2 = self.target_transform(image2)
        
        image_pano = self.normal_transform(self.loader(full_path, self.downscale))
        
        prompt = self.prompt # fixed index from input

        return {
            'img1': image1,
            'img2': image2,
            'img1_original': image1_original,
            'img2_original': image2_original,
            'pano': image_pano,
            'rotation_x1': rotation_x1,
            'rotation_y1': rotation_y1,
            'rotation_x2': rotation_x2,
            'rotation_y2': rotation_y2,
            'path': path,
            'path2': path2,
            'pano_path': full_path,
            'txt':prompt,
        }

    def __len__(self):

        return 5
        
    def init_crops_transform(self):
        transform = transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        target_transform=transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        
        return transform, target_transform
    
    
    def normal_transform(self, image):
        image_arr = np.array(image)
        image_arr = (image_arr / 255).astype(np.float32)
        return torch.tensor(image_arr)
    
    def loader(self, path, down_scale=1):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if down_scale > 1:
            width, height = img.size
            new_width = int(width / down_scale)
            new_height = int(height / down_scale)
            new_size = (new_width, new_height)
            img = img.resize(new_size)
        return img