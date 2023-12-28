import numpy as np
import cv2
import os
import plotly.graph_objs as go
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data_utils import gen_sample_grid
import glob
import json
import math
import sys

def load_rgb(path):
    Pano = np.asarray(cv2.imread(path)).astype(np.float64)
    return Pano

# TODO: modified to save full image
naive_vis = False

if __name__=="__main__":
    root_dir = str(sys.argv[1])
    # split = ["train", "val", "test"]
    pano_num = int(sys.argv[2])
    downscale = 1
    
    out_root = str(sys.argv[4])
    crop_res = [256, 256]
    theta_range = [-15, 15]
    phi_range = [-15, 15]
    crop_num = int(sys.argv[3])
    starting_idx = 0
    
    if naive_vis:
        dist_root = os.path.join(out_root, "dist")
        dist_meta_all = {}
    undist_root = os.path.join(out_root, "undist")
    undist_meta_all = {}
    
    path_dict = sorted([y for x in os.walk(root_dir) for y in glob.glob(os.path.join(x[0], "*.jpg"))])
    
    for pano_idx in tqdm(range(starting_idx, starting_idx+pano_num)):
        thetas = np.random.uniform(low=theta_range[0], high=theta_range[1], size=crop_num)
        phis = np.random.uniform(low=phi_range[0], high=phi_range[1], size=crop_num)
        pano_path = path_dict[pano_idx]
        Pano = load_rgb(pano_path)
        
        Pano_name = (pano_path.split('/')[-1]).split('.')[0]
        if naive_vis:
            pano_dist_root = os.path.join(dist_root, Pano_name)
            os.makedirs(pano_dist_root, exist_ok=True)
            dist_meta = {}
            
        pano_undist_root = os.path.join(undist_root, Pano_name, 'crops')
        os.makedirs(pano_undist_root, exist_ok=True)
        undist_meta = {}
        
        Pano_shape = Pano.shape[:2]
        Pano = torch.from_numpy(Pano[None,:]).permute([0,3,1,2])     
        
        for crop_idx in range(0, crop_num):
            if naive_vis:
                sample_grid = gen_sample_grid(Pano_shape, crop_res, thetas[crop_idx], phis[crop_idx])
                sample_grid = torch.from_numpy(sample_grid[None, :])
                patch = F.grid_sample(Pano, sample_grid, align_corners=True)
                patch = patch[0].numpy().transpose([1,2,0])
            
            undist_sample_grid = gen_sample_grid(Pano_shape, crop_res, thetas[crop_idx], phis[crop_idx], fov=90, undist=True)
            undist_sample_grid = torch.from_numpy(undist_sample_grid[None, :])
            undist_patch = F.grid_sample(Pano, undist_sample_grid, align_corners=True)
            undist_patch = undist_patch[0].numpy().transpose([1,2,0])
            
            x = (thetas[crop_idx] / 180) * math.pi
            y = (phis[crop_idx] / 180) * math.pi
            """
            x range from -pi to pi (theta from -180 to 180)
            y range from -pi/6 to pi/6 (theta from -30 to 30)
            """
            if naive_vis:
                crop_dist_path = os.path.join(pano_dist_root, '{:03}.jpg'.format(crop_idx))
                dist_meta['crop_{:03}'.format(crop_idx)]={'x': x, 'y': y}
                cv2.imwrite(crop_dist_path, patch)
        
            crop_undist_path = os.path.join(pano_undist_root, '{:03}.jpg'.format(crop_idx))
            undist_meta['crop_{:03}'.format(crop_idx)]={'x': x, 'y': y}
            cv2.imwrite(crop_undist_path, undist_patch)
        
        pano_save_path = os.path.join(undist_root, Pano_name, 'panorama.jpg')
        cv2.imwrite(pano_save_path, Pano[0].numpy().transpose([1,2,0]))
        
        if naive_vis:
            dist_meta_all[Pano_name] = dist_meta
        undist_meta_all[Pano_name] = undist_meta

    if naive_vis:
        dist_json_path = "{}/meta.json".format(dist_root)
        with open(dist_json_path, 'w') as out_file:
            json.dump(dist_meta_all, out_file, sort_keys = True, indent = 4, ensure_ascii = False)
    undist_json_path = "{}/meta.json".format(undist_root)
    with open(undist_json_path, 'w') as out_file:
        json.dump(undist_meta_all, out_file, sort_keys = True, indent = 4, ensure_ascii = False)
 