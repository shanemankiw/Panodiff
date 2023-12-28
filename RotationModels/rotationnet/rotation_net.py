import torch
import torch.nn as nn
from RotationModels.utils.compute_utils import *
from RotationModels.preact_resnet import PreActBlock, PreActBottleneck


class RotationNet(nn.Module):
    def __init__(self, cfgmodel):
        super(RotationNet, self).__init__()
        block_type = [PreActBlock, PreActBottleneck]
        block = block_type[cfgmodel.block]
        num_blocks = [int(x) for x in cfgmodel.num_blocks.strip().split("-")]
        self.zdim = cfgmodel.zdim
        if hasattr(cfgmodel, "in_planes"):
            self.in_planes = int(cfgmodel.in_planes)
        else:
            self.in_planes = 2048
        self.out_rotation_mode = cfgmodel.out_rotation_mode
        if (self.out_rotation_mode == "Quaternion"):
            self.out_dim = 4
        elif (self.out_rotation_mode == "ortho6d"):
            self.out_dim = 6
        elif (self.out_rotation_mode == "ortho5d"):
            self.out_dim = 5
        elif (self.out_rotation_mode == "axisAngle"):
            self.out_dim = 4
        elif (self.out_rotation_mode == "euler"):
            self.out_dim = 3
        elif self.out_rotation_mode == "angle":
            self.out_dim = 1
        elif self.out_rotation_mode == "distribution":
            dist_out = getattr(cfgmodel, "out_dim", 360)

        self.layer1 = self._make_layer(block, 512, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 512, num_blocks[1], stride=2)
        self.linear = nn.Linear(512 * block.expansion, self.zdim)
        self.linear2 = nn.Linear(self.zdim, self.out_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        batch = x.shape[0]
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out_rotation = self.linear2(out)

        if (self.out_rotation_mode == "Quaternion"):
            out_r_mat = compute_rotation_matrix_from_quaternion(out_rotation)
        elif (self.out_rotation_mode == "ortho6d"):
            out_r_mat = compute_rotation_matrix_from_ortho6d(out_rotation)
        elif (self.out_rotation_mode == "ortho5d"):
            out_r_mat = compute_rotation_matrix_from_ortho5d(out_rotation)
        elif (self.out_rotation_mode == "axisAngle"):
            out_r_mat = compute_rotation_matrix_from_axisAngle(out_rotation)
        elif (self.out_rotation_mode == "euler"):
            out_r_mat = compute_rotation_matrix_from_euler(out_rotation)
        elif self.out_rotation_mode == "angle":
            out_r_mat = compute_rotation_matrix_from_viewpoint(out_rotation, torch.zeros(out_rotation.size(), requires_grad=True).cuda(),
                                                               out_rotation.size(0))
        elif self.out_rotation_mode == "distribution":
            _, rotation_x = torch.topk(out_rotation, 1, dim=-1)
            out_r_mat = compute_rotation_matrix_from_viewpoint(rotation_x.float()/self.out_dim*2*pi, torch.zeros(rotation_x.size(), requires_grad=True).cuda(),
                                                               rotation_x.size(0))
        return out_r_mat.cuda(), out_rotation
