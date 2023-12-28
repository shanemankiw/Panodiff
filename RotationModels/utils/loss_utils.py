import torch
import torch.nn as nn
from torch import optim
from RotationModels.utils.compute_utils import *
import numpy as np


def get_opt(params, cfgopt):
    if cfgopt.type == 'adam':
        optimizer = optim.Adam(params, lr=float(cfgopt.lr),
                               betas=(cfgopt.beta1, cfgopt.beta2),
                               weight_decay=cfgopt.weight_decay)
    elif cfgopt.type == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=float(cfgopt.lr), momentum=cfgopt.momentum)
    else:
        assert 0, "Optimizer type should be either 'adam' or 'sgd'"

    scheduler = None
    scheduler_type = getattr(cfgopt, "scheduler", None)
    if scheduler_type is not None:
        if scheduler_type == 'exponential':
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
        elif scheduler_type == 'step':
            step_size = int(getattr(cfgopt, "step_epoch", 500))
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay)
        elif scheduler_type == 'linear':
            step_size = int(getattr(cfgopt, "step_epoch", 2000))

            def lambda_rule(ep):
                lr_l = 1.0 - min(1, max(0, ep - 0.5 * step_size) / float(0.45 * step_size)) * (1 - 0.01)
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear' or 'step'"
        return optimizer, scheduler
    else:
        return optimizer


def rotation_loss_class(out_rotation_x, angle_x):
    length = out_rotation_x.size(-1)
    label = ((angle_x.view(-1).cuda() + pi) / 2 / np.pi * length)
    label[label < 0] += length
    label[label >= length] -= length
    if out_rotation_x.size(-1) == 1:
        loss_x = ((out_rotation_x - angle_x.view(-1).cuda()) ** 2).mean()
    elif out_rotation_x.size(-1) == length:
        criterion = nn.CrossEntropyLoss()
        loss_x = criterion(out_rotation_x, label.long())
    else:
        assert False
    return loss_x


def rotation_loss_reg(predict_rotation, gt_rotation):
    l2_loss = ((predict_rotation.view(-1).cuda() - gt_rotation.view(-1).cuda()) ** 2)
    loss = l2_loss.mean()
    res = {
        "loss": loss,
        "rotation_l2_error": l2_loss,
        "rotation_l2_loss": l2_loss.mean(),
    }
    return res
