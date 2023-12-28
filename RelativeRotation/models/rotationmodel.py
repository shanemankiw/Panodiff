import tqdm
import torch
import importlib
import numpy as np
from utils.compute_utils import *

class RotationPredictionModel():
    def __init__(self, cfg):
        self.cfg = cfg
        self.overlap_type = cfg.overlap_type

        encoder_lib = importlib.import_module(cfg.encoder.type)
        self.encoder = encoder_lib.ImageEncoder(cfg.encoder)
        self.encoder.cuda()

        dn_lib = importlib.import_module(cfg.rotationnet.type)
        self.rotation_net = dn_lib.RotationNet(cfg.rotationnet)
        self.rotation_net.cuda()

        dn_lib_y = importlib.import_module(cfg.rotationnet_y.type)
        self.rotation_net_y = dn_lib_y.RotationNet(cfg.rotationnet_y)
        self.rotation_net_y.cuda()

        dn_lib_z = importlib.import_module(cfg.rotationnet_z.type)
        self.rotation_net_z = dn_lib_z.RotationNet(cfg.rotationnet_z)
        self.rotation_net_z.cuda()
    
    def set_eval(self):
        self.encoder.eval()
        self.rotation_net.eval()
        self.rotation_net_y.eval()
        self.rotation_net_z.eval()

    def __call__(self, test_loader):
        print("Validation on relative rotation prediction for {} cases.".format(self.overlap_type))
        
        rot_x, rot_y1, rot_y2= None, None, None
        for data_full in tqdm.tqdm(test_loader):
            img1 = data_full['img1'].cuda()
            img2 = data_full['img2'].cuda()

            image_feature_map1 = self.encoder(img1)
            image_feature_map2 = self.encoder(img2)
            pairwise_feature = compute_correlation_volume_pairwise(
                image_feature_map1, image_feature_map2, num_levels=1
            )
            
            _, out_rotation_x = self.rotation_net(pairwise_feature)
            _, out_rotation_y = self.rotation_net_y(pairwise_feature)
            _, out_rotation_z = self.rotation_net_z(pairwise_feature)
            
            _, rotation_x = torch.topk(out_rotation_x, 1, dim=-1)
            _, rotation_y = torch.topk(out_rotation_y, 1, dim=-1)
            _, rotation_z = torch.topk(out_rotation_z, 1, dim=-1)
            
            rot_x = rotation_x if rot_x is None else torch.cat((rot_x, rotation_x))
            rot_y1 = rotation_y if rot_y1 is None else torch.cat((rot_y1, rotation_y))
            rot_y2 = rotation_z if rot_y2 is None else torch.cat((rot_y2, rotation_z))

        ret_dict={}
        ret_dict['rot_x']  = rot_x[:, 0]
        ret_dict['rot_y1'] = rot_y1[:, 0]
        ret_dict['rot_y2'] = rot_y2[:, 0]
        return ret_dict

    def load(self, path, strict=True):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        self.rotation_net.load_state_dict(ckpt['dn'], strict=strict)
        self.rotation_net_y.load_state_dict(ckpt['dny'], strict=strict)
        self.rotation_net_z.load_state_dict(ckpt['dnz'], strict=strict)
