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


if __name__=="__main__":
    dataset_dir = "/data/chenziyu/myprojects/PanoData"
    root_dir = "/data/chenziyu/myprojects/PanoData/my_sun360"

    # total train num 34260
    train_num = 34260
    downscale = 1
    
    path_dict = sorted([y for x in os.walk(root_dir) for y in glob.glob(os.path.join(x[0], "*.jpg"))])
    
    train_data = {}
    
    # for BLIP2
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
    model.to("cuda")
    # the model is downloaded into:
    # ./.cache/huggingface/hub/models--Salesforce--blip2-opt-2.7b/snapshots/ca9e6c21b0ae56818ab19c8c873eb1bb5cfae2f8/vocab.json
    for idx in tqdm(range(0, train_num)):
        try:
            image = load_rgb(path_dict[idx], downscale=downscale)
            # BLIP-2
            image_BLIP = Image.open(path_dict[idx])
            inputs = processor(images=image_BLIP, return_tensors="pt").to("cuda", torch.float16)
            generated_ids = model.generate(**inputs)
            generated_prompt = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            Pano_name = (path_dict[idx].split('/')[-1]).split('.')[0]
            train_data[Pano_name] = generated_prompt            
        except:
            continue
        
    train_json_path = "/data/chenziyu/myprojects/PanoData/my_sun360_prompts.json".format()
    with open(train_json_path, 'w') as out_file:
        json.dump(train_data, out_file, sort_keys = True, indent = 4, ensure_ascii = False)