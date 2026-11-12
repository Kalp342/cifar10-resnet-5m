"""CIFAR pre-activation Wide ResNet; original implementation following Zagoruyko 2016."""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride, dropout):
        super().__init__()
        self.bn1=nn.BatchNorm2d(cin); self.conv1=nn.Conv2d(cin,cout,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(cout);self.conv2=nn.Conv2d(cout,cout,3,1,1,bias=False)
        self.shortcut=nn.Conv2d(cin,cout,1,stride,bias=False) if cin!=cout or stride!=1 else None
        self.dropout=dropout

    def forward(self,x):
        z=F.relu(self.bn1(x),inplace=False)
        skip=x if self.shortcut is None else self.shortcut(z)
        z=self.conv1(z);z=F.relu(self.bn2(z),inplace=True)
        z=F.dropout(z,p=self.dropout,training=self.training)
        return skip+self.conv2(z)


class WideResNet(nn.Module):
    def __init__(self, depth=28, width=3.5, dropout=0.0, classes=10):
        super().__init__()
        assert (depth-4)%6==0 and depth>=10
        widths=[16]+[int(c*width) for c in (16,32,64)];n=(depth-4)//6
        self.stem=nn.Conv2d(3,16,3,1,1,bias=False)
        blocks=[];cin=16
        for stage,cout in enumerate(widths[1:]):
            for b in range(n):
                blocks.append(Block(cin,cout,2 if stage>0 and b==0 else 1,dropout));cin=cout
        self.blocks=nn.Sequential(*blocks);self.bn=nn.BatchNorm2d(cin);self.head=nn.Linear(cin,classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.Linear):nn.init.zeros_(m.bias)
        enforce_budget(self)

    def forward(self,x):
        x=self.blocks(self.stem(x));x=F.relu(self.bn(x),inplace=True)
        return self.head(x.mean((2,3)))


def enforce_budget(model):
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total>5_000_000 or trainable>5_000_000:raise ValueError(f'Parameter budget exceeded: total={total}, trainable={trainable}')
    return {'total':total,'trainable':trainable}


def build_model(config):
    if config.get('architecture')=='se_resnet164':
        return SEResNet164(config.get('base_channels',24))
    return WideResNet(config.get('depth',28),config.get('width',3.5),config.get('dropout',0.0))


class SEBottleneck(nn.Module):
    """Pre-activation bottleneck with SE residual gating (He 2016; Hu 2018)."""
    def __init__(self,cin,planes,stride):
        super().__init__();cout=planes*4;hidden=max(cout//16,1)
        self.bn1=nn.BatchNorm2d(cin);self.c1=nn.Conv2d(cin,planes,1,bias=False)
        self.bn2=nn.BatchNorm2d(planes);self.c2=nn.Conv2d(planes,planes,3,stride,1,bias=False)
        self.bn3=nn.BatchNorm2d(planes);self.c3=nn.Conv2d(planes,cout,1,bias=False)
        self.skip=nn.Conv2d(cin,cout,1,stride,bias=False) if cin!=cout or stride!=1 else None
        self.se=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(cout,hidden,1),nn.ReLU(),nn.Conv2d(hidden,cout,1),nn.Sigmoid())

    def forward(self,x):
        z=F.relu(self.bn1(x));skip=x if self.skip is None else self.skip(z)
        z=self.c1(z);z=self.c2(F.relu(self.bn2(z)));z=self.c3(F.relu(self.bn3(z)))
        return skip+z*self.se(z)


class SEResNet164(nn.Module):
    """CIFAR 9n+2=164, n=18, widened bottlenecks, original implementation."""
    def __init__(self,base_channels=24):
        super().__init__();cin=base_channels
        self.stem=nn.Conv2d(3,cin,3,1,1,bias=False);blocks=[]
        for stage,planes in enumerate([base_channels,base_channels*2,base_channels*4]):
            for index in range(18):
                blocks.append(SEBottleneck(cin,planes,2 if stage and index==0 else 1));cin=planes*4
        self.blocks=nn.Sequential(*blocks);self.bn=nn.BatchNorm2d(cin);self.head=nn.Linear(cin,10)
        for module in self.modules():
            if isinstance(module,nn.Conv2d):nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(module,nn.Linear):nn.init.zeros_(module.bias)
        enforce_budget(self)

    def forward(self,x):
        x=self.blocks(self.stem(x));return self.head(F.relu(self.bn(x)).mean((2,3)))
