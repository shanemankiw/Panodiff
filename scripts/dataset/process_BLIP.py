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

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

def load_rgb(path, downscale=1):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if downscale != 1:
        img = rescale(img, 1./downscale, anti_aliasing=False, channel_axis=-1)
        
    return img


    dataset_dir = "/data/chenziyu/myprojects/PanoData"
    root_dir = "/data/chenziyu/myprojects/PanoData/my_sun360"
    mask_path = "/data/chenziyu/myprojects/OmniDreamer/assets/90binarymask.png"
    split = ["train", "val", "test"]

    '''train_num = int(sys.argv[2])
    test_num = int(train_num/10)
    downscale = int(sys.argv[1]) # source resolution 1024X512'''
    # total train num 34260
    train_num = 33000 
    test_num = 1260
    downscale = 2


    out_dir = "{}/BLIP_sun360_d{:01}_t{:05}_v{:05}".format(dataset_dir, downscale, train_num, test_num)
    
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

    # for BLIP2
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
    model.to("cuda")
    # the model is downloaded into:
    # ./.cache/huggingface/hub/models--Salesforce--blip2-opt-2.7b/snapshots/ca9e6c21b0ae56818ab19c8c873eb1bb5cfae2f8/vocab.json
    a = 1
    for idx in tqdm(range(0, train_num+test_num)):
        try:
            image = load_rgb(path_dict[idx], downscale=downscale)
            # BLIP-2
            image_BLIP = Image.open(path_dict[idx])
            inputs = processor(images=image_BLIP, return_tensors="pt").to("cuda", torch.float16)
            generated_ids = model.generate(**inputs)
            generated_prompt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            source = np.multiply(image, mask)
            
            target = np.clip(image*255, 0, 255).astype(np.uint8)
            source = np.clip(source*255, 0, 255).astype(np.uint8)
            
            
            if idx < train_num:
                source_path = "{}/train_{:05}.jpg".format(source_dir, idx)
                target_path = "{}/train_{:05}.jpg".format(target_dir, idx)
                image_dict = {
                    "source": "{}/train_{:05}.jpg".format("source", idx),
                    "target": "{}/train_{:05}.jpg".format("target", idx),
                    "prompt": generated_prompt
                }
                train_data.append(image_dict)
            elif idx>=train_num and idx<(train_num + test_num):
                source_path = "{}/test_{:05}.jpg".format(source_dir, idx)
                target_path = "{}/test_{:05}.jpg".format(target_dir, idx)
                image_dict = {
                    "source": "{}/test_{:05}.jpg".format("source", idx),
                    "target": "{}/test_{:05}.jpg".format("target", idx),
                    "prompt": generated_prompt
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
 