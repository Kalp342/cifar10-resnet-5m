"""
Model definitions for the CIFAR-10 ResNet checkpoints (NYU Tandon DL F26, team kpp10037).

Every checkpoint is a plain PyTorch state_dict of the `ResNet` class below. All of them use the same
block design and differ only in stage widths / blocks per stage:

  checkpoint                         widths          blocks   params      role
  teacher_b11_model_final.pt         128,256,512     3,4,3    18,470,538  teacher (seed 11, 200 epochs)
  teacher_b12_model_final.pt         128,256,512     3,4,3    18,470,538  teacher (seed 12, 200 epochs)
  teacher_w1_model_final.pt          96,192,384      3,4,3    10,393,834  teacher (seed 1, 160 epochs)
  teacher_w2_model_final.pt          96,192,384      3,4,3    10,393,834  teacher (seed 2, 160 epochs)
  kd2_student_model_final.pt         64,128,256      3,5,3     4,918,602  student, Kaggle public 0.904 (10-view TTA)
  soup_kd2_model_final.pt            64,128,256      3,5,3     4,918,602  model soup of kd2, Kaggle public 0.905 (10-view TTA)

All were trained from scratch on the 50,000 competition training images only (no pretrained weights,
no external data). Weights are the EMA (decay 0.999) weights; the soup is the uniform average of 4
fine-tuned copies of kd2_student (BatchNorm running stats averaged too).

Input pipeline (identical for every checkpoint):
  RGB uint8 image (32x32x3)  ->  float / 255  ->  (x - MEAN) / STD per channel  ->  NCHW tensor
MEAN / STD below are the per-channel statistics of the 50k training images (they are NOT stored in the
checkpoints).

Usage:
  from model import load_model, predict
  net = load_model("soup_kd2_model_final.pt")             # architecture chosen from the file name
  probs = predict(net, images_uint8_nhwc, views=10)         # (N, 10) softmax probabilities
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# per-channel statistics of the 50,000 training images, on the 0-1 scale (R, G, B)
MEAN = (0.49139967861519607, 0.48215840839460783, 0.44653091444546567)
STD = (0.24703223246174102, 0.24348512800151828, 0.26158784172803257)

# outputs are class indices 0-9 in the competition's train_labels.csv numbering


class BasicBlock(nn.Module):
    """y = ReLU(S(x) + F(x)); F = conv3x3-BN-ReLU-conv3x3-BN, S = identity or 1x1 conv-BN projection."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet(nn.Module):
    """3-stage ResNet for 32x32 inputs: stem conv3x3 -> stages of BasicBlocks -> global avg pool -> FC.
    The first block of each stage uses strides[i] (default 1,2,2: feature maps 32x32 -> 16x16 -> 8x8)."""
    def __init__(self, widths=(64, 128, 256), blocks=(3, 4, 3), num_classes=10, strides=(1, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(BasicBlock(cin, w, strides[i] if j == 0 else 1)); cin = w
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# architecture of each released checkpoint
CONFIGS = {
    "teacher_b11": dict(widths=(128, 256, 512), blocks=(3, 4, 3)),
    "teacher_b12": dict(widths=(128, 256, 512), blocks=(3, 4, 3)),
    "teacher_w1": dict(widths=(96, 192, 384), blocks=(3, 4, 3)),
    "teacher_w2": dict(widths=(96, 192, 384), blocks=(3, 4, 3)),
    "kd2_student": dict(widths=(64, 128, 256), blocks=(3, 5, 3)),
    "soup_kd2": dict(widths=(64, 128, 256), blocks=(3, 5, 3)),
}


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_model(path, name=None, device="cpu"):
    """Build the right ResNet for a checkpoint and load its weights (eval mode).
    `name` is a CONFIGS key; if omitted it is taken from the file name (e.g. 'teacher_b11_model_final.pt')."""
    if name is None:
        name = next((k for k in CONFIGS if os.path.basename(path).startswith(k)), None)
        if name is None:
            raise ValueError(f"can't infer architecture from {path!r}; pass name= one of {list(CONFIGS)}")
    net = ResNet(**CONFIGS[name])
    sd = torch.load(path, map_location="cpu")
    # EMA checkpoints saved from torch.optim.swa_utils.AveragedModel carry a 'module.' prefix and an
    # 'n_averaged' counter; strip both so the keys match the plain model
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items() if k != "n_averaged"}
    net.load_state_dict(sd)
    return net.to(device).eval()


@torch.no_grad()
def predict(net, images, views=10, batch_size=500, device=None):
    """Softmax probabilities for uint8 RGB images shaped (N, 32, 32, 3).
    views=1: plain; views=2: + horizontal flip; views=10: flip x {centre, +-2 px shifts in x and y}
    (reflect padding), the test-time augmentation used for the Kaggle submissions."""
    device = device or next(net.parameters()).device
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    shifts = [(2, 2)] if views in (1, 2) else [(2, 2), (0, 2), (4, 2), (2, 0), (2, 4)]
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(images[i:i + batch_size])).to(device)
        x = (x.permute(0, 3, 1, 2).float() / 255 - mean) / std
        xp = F.pad(x, (2, 2, 2, 2), mode="reflect")
        p = 0
        for dy, dx in shifts:
            v = xp[:, :, dy:dy + 32, dx:dx + 32]
            p = p + net(v).softmax(1)
            if views != 1:
                p = p + net(v.flip(3)).softmax(1)
        out.append((p / p.sum(1, keepdim=True)).cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    # sanity check: build every architecture and print its parameter count
    for k, cfg in CONFIGS.items():
        print(f"{k:12s} {count_params(ResNet(**cfg)):>11,} params")
