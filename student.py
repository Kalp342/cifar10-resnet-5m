"""Submitted architecture and CPU evaluator extracted from the competition run."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PostActBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        if cin != cout or stride != 1:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        else:
            self.short = nn.Identity()

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)), inplace=True)
        z = self.bn2(self.conv2(z))
        return F.relu(z + self.short(x), inplace=True)


class TeammateResNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        blocks, cin = [], 64
        for width, depth, stride in [(64, 3, 1), (128, 5, 2), (256, 3, 2)]:
            for index in range(depth):
                blocks.append(PostActBlock(cin, width, stride if index == 0 else 1))
                cin = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(cin, classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, PostActBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean((2, 3)))


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total > 5_000_000 or trainable > 5_000_000:
        raise ValueError(f'budget exceeded: {total}')
    return {'total': total, 'trainable': trainable}


@torch.inference_mode()
def probs(model, pixels_nhwc, stats, batch=250):
    mean = torch.tensor(stats['mean'])[None, :, None, None]
    std = torch.tensor(stats['std'])[None, :, None, None]
    out = []
    for i in range(0, len(pixels_nhwc), batch):
        x = torch.from_numpy(pixels_nhwc[i:i + batch]).permute(0, 3, 1, 2).float() / 255
        x = (x - mean) / std
        p = (model(x).float().softmax(1) + model(x.flip(-1)).float().softmax(1)) * 0.5
        out.append(p)
    return torch.cat(out).numpy()
