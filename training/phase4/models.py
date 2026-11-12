"""Original implementations following Gastaldi/hysts; see REFERENCES.md."""
import torch
from torch import nn
import torch.nn.functional as F
from src.models import Block,WideResNet

class ShakeMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,alpha,beta):
        ctx.save_for_backward(beta)
        return alpha*a+(1-alpha)*b
    @staticmethod
    def backward(ctx,gradient):
        beta,=ctx.saved_tensors
        return beta*gradient,(1-beta)*gradient,None,None

class Branch(nn.Module):
    def __init__(self,cin,cout,stride):
        super().__init__()
        self.conv1=nn.Conv2d(cin,cout,3,stride,1,bias=False);self.bn1=nn.BatchNorm2d(cout)
        self.conv2=nn.Conv2d(cout,cout,3,1,1,bias=False);self.bn2=nn.BatchNorm2d(cout)
    def forward(self,x):
        return self.bn2(self.conv2(F.relu(self.bn1(self.conv1(F.relu(x))),inplace=False)))

class ShiftSkip(nn.Module):
    def __init__(self,cin,cout,stride):
        super().__init__();self.stride=stride
        self.path1=nn.Conv2d(cin,cout//2,1,bias=False);self.path2=nn.Conv2d(cin,cout//2,1,bias=False);self.bn=nn.BatchNorm2d(cout)
    def forward(self,x):
        x=F.relu(x);other=F.pad(x[:,:,1:,1:],(0,1,0,1))
        return self.bn(torch.cat([self.path1(F.avg_pool2d(x,1,self.stride)),self.path2(F.avg_pool2d(other,1,self.stride))],1))

class ShakeBlock(nn.Module):
    def __init__(self,cin,cout,stride):
        super().__init__();self.left=Branch(cin,cout,stride);self.right=Branch(cin,cout,stride)
        self.skip=ShiftSkip(cin,cout,stride) if cin!=cout or stride!=1 else nn.Identity()
    def forward(self,x):
        a=self.left(x);b=self.right(x)
        if self.training:
            # Independent Uniform[0,1] coefficients per image, block and pass.
            alpha=torch.rand((len(x),1,1,1),device=x.device,dtype=a.dtype)
            beta=torch.rand_like(alpha)
            mixed=ShakeMix.apply(a,b,alpha,beta)
        else:mixed=(a+b)*.5
        return self.skip(x)+mixed

class ShakeShake26(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Conv2d(3,16,3,1,1,bias=False);self.bn=nn.BatchNorm2d(16)
        blocks=[];cin=16
        for stage,cout in enumerate([32,64,128]):
            for index in range(4):
                blocks.append(ShakeBlock(cin,cout,2 if stage and index==0 else 1));cin=cout
        self.blocks=nn.Sequential(*blocks);self.head=nn.Linear(128,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m,nn.Linear):nn.init.zeros_(m.bias)
    def forward(self,x):return self.head(F.relu(self.blocks(self.bn(self.stem(x)))).mean((2,3)))

class LargeWRN(nn.Module):
    """Same CIFAR WRN blocks as A, wider; research-only teacher exception."""
    def __init__(self,width=10,dropout=0.):
        super().__init__();self.stem=nn.Conv2d(3,16,3,1,1,bias=False);blocks=[];cin=16
        for stage,cout in enumerate([int(16*width),int(32*width),int(64*width)]):
            for i in range(4):blocks.append(Block(cin,cout,2 if stage and i==0 else 1,dropout));cin=cout
        self.blocks=nn.Sequential(*blocks);self.bn=nn.BatchNorm2d(cin);self.head=nn.Linear(cin,10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.Linear):nn.init.zeros_(m.bias)
    def forward(self,x):return self.head(F.relu(self.bn(self.blocks(self.stem(x)))).mean((2,3)))

def counts(model,teacher=False):
    result={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if not teacher and (result['total']>=5_000_000 or result['trainable']>=5_000_000):raise ValueError(result)
    return result

def build(cfg):
    arch=cfg['architecture'];teacher=cfg.get('role')=='teacher'
    if arch=='wrn28_10':
        if not teacher:raise ValueError('Oversized WRN may only be a research teacher')
        model=LargeWRN(10,cfg.get('dropout',0.))
    elif arch=='shake26_2x32':model=ShakeShake26()
    elif arch=='wrn28_35':model=WideResNet(28,3.5,0.)
    else:raise ValueError(arch)
    counts(model,teacher);return model
