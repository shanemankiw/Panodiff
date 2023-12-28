import os
import tqdm
import torch
import importlib
from utils.compute_utils import *

class OverlapClassificationModel():
    def __init__(self, cfg):
        self.cfg = cfg
                
        encoder_lib = importlib.import_module(cfg.encoder.type)
        self.encoder = encoder_lib.ImageEncoder(cfg.encoder)
        self.encoder.cuda()
        
        classifier_lib = importlib.import_module(cfg.classifier.type)
        self.img_classifier = classifier_lib.RotationNet(cfg.classifier)
        self.img_classifier.cuda()
    
    def set_eval(self):
        self.encoder.eval()
        self.img_classifier.eval()

    def __call__(self, test_loader):
        print("Validation on overlap classification")
        pred_all = None

        for data_full in tqdm.tqdm(test_loader):
            img1 = data_full['img1'].cuda()
            img2 = data_full['img2'].cuda()
            labels = data_full['overlap_status'].cuda()
            
            image1_feature = self.encoder(img1)
            image2_feature = self.encoder(img2)
            
            pairwise_feature = compute_correlation_volume_pairwise(image1_feature, image2_feature, num_levels=1)
            outputs = self.img_classifier(pairwise_feature)
            
            _, predicted_labels = torch.topk(outputs, 1, dim=-1)
            predicted_labels = predicted_labels.view(-1)
            
            pred_all = predicted_labels if pred_all is None else torch.cat([pred_all, predicted_labels], dim=0)
        return pred_all

    def load(self, path, strict=True):
        ckpt = torch.load(path)
        self.img_classifier.load_state_dict(ckpt['classifier'], strict=strict)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
