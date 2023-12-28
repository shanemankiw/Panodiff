from tqdm import tqdm
import numpy as np
import os
import json
import sys

"""
metadata standard structure:
meta (dict)={1:
    {'img1': {'path': '', 'x': , 'y': },
    'img2': {'path': '', 'x': , 'y': },
    'translation':},
    ...
}
x range from -pi to pi (theta from -180 to 180)
y range from -pi/6 to pi/6 (theta from -30 to 30)
translation: always 0
""" 

if __name__=="__main__":
    crop_num = int(sys.argv[1])
    pairs_num = int(sys.argv[2])
    crops_img_path = str(sys.argv[3])
    crop_metadata_path = os.path.join(crops_img_path, 'meta.json')
    
    raw_meta = json.load(open(crop_metadata_path, 'r'))
    n_Pano = len(raw_meta)
    pairs_meta={}
    pair_cnt = 0
    for pano_name in tqdm(raw_meta.keys()):
        crops_dict = raw_meta[pano_name]
        pano_base_dir = os.path.join(crops_img_path, pano_name, "crops")
        
        img1_idxs = np.random.randint(low=0, high=crop_num, size=pairs_num)
        offsets = np.random.randint(low=1, high=crop_num-1, size=pairs_num)
        img2_idxs = np.mod(img1_idxs + offsets, 50)
        
        for pair_idx in range(0, pairs_num):
            img1_idx = "crop_{:03}".format(img1_idxs[pair_idx])
            img2_idx = "crop_{:03}".format(img2_idxs[pair_idx])
            
            img1_dict = crops_dict[img1_idx]
            img2_dict = crops_dict[img2_idx]
            
            if img1_dict["x"] > img2_dict["x"]:
                is_overlap = (img1_dict["x"] - img2_dict["x"]) < np.pi / 2
            else:
                is_overlap = (img2_dict["x"] - img1_dict["x"]) < np.pi / 2
                
            one_pair = {
                'pano':
                    {
                        'path': os.path.join('{}/{}.jpg'.format(pano_name, 'panorama')),
                        'idx': pano_name
                    },
                'img1':
                    {
                        'path': os.path.join('{}/crops/{:03}.jpg'.format(pano_name, img1_idxs[pair_idx])),
                        'x': img1_dict["x"],
                        'y': img1_dict["y"]
                    },
                'img2':
                    {
                        'path': os.path.join('{}/crops/{:03}.jpg'.format(pano_name, img2_idxs[pair_idx])),
                        'x': img2_dict["x"],
                        'y': img2_dict["y"]
                    },
                'translation': 0., 
                'is_overlap': is_overlap,
            }
            pairs_meta[pair_cnt] = one_pair
            
            pair_cnt += 1
            
    np.save('sun360_example.npy', pairs_meta, allow_pickle=True)