import os
import numpy as np
import cv2
import glob
import imageio
import json
import sys
import skimage
from skimage.transform import rescale
from tqdm import tqdm


def load_rgb(path, downscale=1):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if downscale != 1:
        img = rescale(img, 1./downscale, anti_aliasing=False, channel_axis=-1)
        
    # NOTE: pixel values between [-1,1]
    # img -= 0.5
    # img *= 2.
    # img = img.transpose(2, 0, 1)
    return img


if __name__=="__main__":
    root_dir = "/data/chenziyu/myprojects/ControlNet/datasets/SUN360"
    mask_path = "/data/chenziyu/myprojects/OmniDreamer/assets/90binarymask.png"
    split = ["train", "val", "test"]
    train_num = int(sys.argv[2])
    test_num = int(train_num/10)
    downscale = int(sys.argv[1]) # source resolution 1024X512
    prompt = "panoramic"
    
    out_dir = "/data/chenziyu/myprojects/ControlNet/datasets/sun360_d{:01}_t{:05}_v{:05}".format(downscale, train_num, test_num)
    
    source_dir = os.path.join(out_dir, "source")
    target_dir = os.path.join(out_dir, "target")
    
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    
    path_dict = sorted([y for x in os.walk(root_dir) for y in glob.glob(os.path.join(x[0], "*.jpg"))])
    
    mask = load_rgb(mask_path, downscale=downscale)
    
    train_data = []
    test_data = []
    # import ipdb
    # ipdb.set_trace(context=15)
    
    for idx in tqdm(range(0, train_num+test_num)):
        try:
            image = load_rgb(path_dict[idx], downscale=downscale)
            source = np.multiply(image, mask)
            
            target = np.clip(image*255, 0, 255).astype(np.uint8)
            source = np.clip(source*255, 0, 255).astype(np.uint8)
            
            
            if idx < train_num:
                source_path = "{}/train_{:05}.jpg".format(source_dir, idx)
                target_path = "{}/train_{:05}.jpg".format(target_dir, idx)
                image_dict = {
                    "source": "{}/train_{:05}.jpg".format("source", idx),
                    "target": "{}/train_{:05}.jpg".format("target", idx),
                    "prompt": prompt
                }
                train_data.append(image_dict)
            elif idx>=train_num and idx<(train_num + test_num):
                source_path = "{}/test_{:05}.jpg".format(source_dir, idx)
                target_path = "{}/test_{:05}.jpg".format(target_dir, idx)
                image_dict = {
                    "source": "{}/test_{:05}.jpg".format("source", idx),
                    "target": "{}/test_{:05}.jpg".format("target", idx),
                    "prompt": prompt
                }
                test_data.append(image_dict)
            else:
                break
            cv2.imwrite(source_path, cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
            cv2.imwrite(target_path, cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
        except:
            continue
        
    train_json_path = "{}/train.json".format(out_dir)
    test_json_path = "{}/test.json".format(out_dir)
    with open(train_json_path, 'w') as out_file:
        json.dump(train_data, out_file, sort_keys = True, indent = 4, ensure_ascii = False)
    with open(test_json_path, 'w') as out_file:
        json.dump(test_data, out_file, sort_keys = True, indent = 4, ensure_ascii = False)
 