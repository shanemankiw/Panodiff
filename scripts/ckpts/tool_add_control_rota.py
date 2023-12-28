import sys
import os

#assert len(sys.argv) == 3, 'Args are wrong.'

#input_path = sys.argv[1]
#output_path = sys.argv[2]
# HACK
sys.path.append('/HDD/22Ubuntu/code/PanoDiff')
diffusion_model_path = 'pretrained_models/sd-v1-5-inpainting.ckpt'
rotation_model_path = 'pretrained_models/rota/sun360_cv_distribution.pt'

output_path = 'pretrained_models/processed/rota_control_sd.ckpt'

assert os.path.exists(diffusion_model_path), 'Diffusion model does not exist.'
assert os.path.exists(rotation_model_path), 'Diffusion model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import *
from cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./models/rota_inpaint.yaml')
'''
NOTE (Done) 1.1 
Load the whole model, along with the rotaNet.
But at the same time, load the rotaNet weights.
'''
sd_pretrained_weights = torch.load(diffusion_model_path)
rota_pretrained_weights = torch.load(rotation_model_path)

if 'state_dict' in sd_pretrained_weights:
    sd_pretrained_weights = sd_pretrained_weights['state_dict']

if 'state_dict' in rota_pretrained_weights:
    rota_pretrained_weights = rota_pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    is_rotation, name_r = get_node_name(k, 'rotation_model.')
    
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    
    if copy_k in sd_pretrained_weights:
        target_dict[k] = sd_pretrained_weights[copy_k].clone()
    elif is_rotation:
        if name_r.split('.')[0] == 'encoder':
            target_dict[k] = rota_pretrained_weights['enc'][name_r[8:]].clone()
        elif name_r.split('.')[0] == 'rotation_net':
            target_dict[k] = rota_pretrained_weights['dn'][name_r[13:]].clone()
        elif name_r.split('.')[0] == 'rotation_net_y':
            target_dict[k] = rota_pretrained_weights['dny'][name_r[15:]].clone()
        elif name_r.split('.')[0] == 'rotation_net_z':
            target_dict[k] = rota_pretrained_weights['dnz'][name_r[15:]].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
