import torch.nn as nn
import torch.nn.functional as F
from RotationModels.preact_resnet import PreActBlock, PreActBottleneck

class ImageEncoder(nn.Module):
    def __init__(self, cfgmodel):
        super(ImageEncoder, self).__init__()
        block_type = [PreActBlock, PreActBottleneck]
        block = block_type[cfgmodel.block]
        num_blocks = [int(x) for x in cfgmodel.num_blocks.strip().split("-")]
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # todo recheck
        out = self.conv1(x)
        out = self.layer1(out)
        out = F.avg_pool2d(out, 2)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 2)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 2)
        return out
