import os
import yaml
import torch
import argparse
import numpy as np

from math import pi
from tqdm import tqdm
from torch.backends import cudnn
from dataset import get_test_loader
from models import OverlapClassificationModel, RotationPredictionModel
from utils.compute_utils import *

def get_args():
    # command line args
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('config', type=str,
                        help='The configuration file.')
    parser.add_argument('--save_path', type=str, default='./pred_results.npy',
                        help='The path to save the prediction results.')

    # distributed training
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')

    # Resume:
    parser.add_argument('--classification_model_path', default=None, type=str,
                        help="Pretrained cehckpoint for classification")
    parser.add_argument('--overlap_regression_model_path', default=None, type=str,
                        help="Pretrained cehckpoint for overlap regression")
    parser.add_argument('--nonoverlap_regression_model_path', default=None, type=str,
                        help="Pretrained cehckpoint for nonoverlap regression")
    parser.add_argument('--val_angle', default=True, action='store_true',
                        help="Evaluate yaw and pitch error")
    args = parser.parse_args()

    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace

    # parse config file

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return args, config


def main_worker(cfg, args):
    # basic setup
    cudnn.benchmark = True

    _, test_loader = get_test_loader(cfg.data)
    gt_metadata = np.load(cfg.data.pairs_file, allow_pickle=True).item()
    
    overlap_classification_model = OverlapClassificationModel(cfg.models.overlap_classification)
    overlap_classification_model.load(args.classification_model_path)
    overlap_classification_model.set_eval()
    
    overlap_regression_model = RotationPredictionModel(cfg.models.overlap_rotation_regression_model)
    overlap_regression_model.load(args.overlap_regression_model_path)
    overlap_regression_model.set_eval()
    
    nonoverlap_regression_model = RotationPredictionModel(cfg.models.nonoverlap_rotation_regression_model)
    nonoverlap_regression_model.load(args.nonoverlap_regression_model_path)
    nonoverlap_regression_model.set_eval()
    # run validation
    with torch.no_grad():
        pred_overlap_status = overlap_classification_model(test_loader)
        overlap_rots = overlap_regression_model(test_loader)
        nonelap_rots = nonoverlap_regression_model(test_loader)
    # collect results according to overlap status
    pred_overlap_status = pred_overlap_status.cpu()
    rot_x = overlap_rots['rot_x'].cpu() * pred_overlap_status + nonelap_rots['rot_x'].cpu() * (1 - pred_overlap_status)
    rot_y1 = overlap_rots['rot_y1'].cpu() * pred_overlap_status + nonelap_rots['rot_y1'].cpu() * (1 - pred_overlap_status)
    rot_y2 = overlap_rots['rot_y2'].cpu() * pred_overlap_status + nonelap_rots['rot_y2'].cpu() * (1 - pred_overlap_status)
    
    # compute metrics
    print("Computing metrics...")
    if args.val_angle:
        gt_rmat_array = None
        out_rmat_array = None
        overlap_all = None
        all_res = {}
        for index in tqdm(range(len(gt_metadata))):
            rotation_x1 = torch.tensor(gt_metadata[index]['img1']['x'])[..., None]
            rotation_x2 = torch.tensor(gt_metadata[index]['img2']['x'])[..., None]
            rotation_y1 = torch.tensor(gt_metadata[index]['img1']['y'])[..., None]
            rotation_y2 = torch.tensor(gt_metadata[index]['img2']['y'])[..., None]
            overlap_status = torch.tensor(gt_metadata[index]['is_overlap'])[..., None].bool()
            
            gt_rmat = compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, 1)
            
            pred_delta_x = rot_x[index][..., None]
            pred_rot_y1 = rot_y1[index][..., None]
            pred_rot_y2 = rot_y2[index][..., None]
            
            rt1 = compute_rotation_matrix_from_viewpoint(
                torch.zeros_like(pred_delta_x), pred_rot_y1.float() / 180 * pi - pi, 1
            ).view(1, 3, 3)
            rt2 = compute_rotation_matrix_from_viewpoint(
                pred_delta_x.float() / 180 * pi - pi, pred_rot_y2.float() / 180 * pi - pi, 1
            ).view(1, 3, 3)
            out_rmat = compute_rotation_matrix_from_two_matrices(
                rt2, rt1
            ).view(1, 3, 3).cuda()
            
            gt_rmat_array = gt_rmat if gt_rmat_array is None else torch.cat((gt_rmat_array, gt_rmat))
            out_rmat_array = out_rmat if out_rmat_array is None else torch.cat((out_rmat_array, out_rmat))
            overlap_all = overlap_status if overlap_all is None else torch.cat((overlap_all, overlap_status))
        
        geodesic_loss = compute_geodesic_distance_from_two_matrices(
            out_rmat_array.view(-1, 3, 3), gt_rmat_array.view(-1, 3, 3)
        ) / pi * 180
        gt_distance = compute_angle_from_r_matrices(gt_rmat_array.view(-1, 3, 3))

        geodesic_loss_overlap = geodesic_loss[overlap_all]
        geodesic_loss_widebaseline = geodesic_loss[~overlap_all]

        res_error = {
            "gt_angle": gt_distance / pi * 180,
            "rotation_geodesic_error_overlap": geodesic_loss_overlap,
            "rotation_geodesic_error_widebaseline": geodesic_loss_widebaseline,
            "rotation_geodesic_error": geodesic_loss,
        }
        for k, v in res_error.items():
            v = v.view(-1).detach().cpu().numpy()
            if k == "gt_angle" or v.size == 0:
                continue
            mean = np.mean(v)
            count_10 = (v <= 10).sum(axis=0)
            percent_10 = np.true_divide(count_10, v.shape[0])
            all_res.update({k + '/mean': mean,
                            k + '/10deg': percent_10*100})
        
        for k, v in all_res.items():
            print(k, ': ', v)
    
    # save for diffusion models
    rot_x = (rot_x - 180) / 180 * torch.pi
    rot_y1 = (rot_y1 - 180) / 180 * torch.pi
    rot_y2 = (rot_y2 - 180) / 180 * torch.pi

    pred_meta = {}
    for index in range(0, len(gt_metadata)):
        one_pair = {}
        one_pair.update(gt_metadata[index])
        
        one_pair['img1']['y'] = rot_y1[index].item()
        one_pair['img2']['y'] = rot_y2[index].item()
        img2_x = one_pair['img1']['x'] + rot_x[index].item()
        if img2_x >= pi:
            img2_x -= 2 * pi
        elif img2_x < -pi:
            img2_x += 2 * pi
        else:
            img2_x = img2_x
        one_pair['img2']['x'] = img2_x
        pred_meta[index] = one_pair
    
    np.save(args.save_path, pred_meta, allow_pickle=True)


def main():
    # command line args
    args, cfg = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    main_worker(cfg, args)


if __name__ == '__main__':
    main()
