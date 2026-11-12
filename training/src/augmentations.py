"""Batch GPU crop/flip and optional evidence-driven color augmentation.

Advanced policies sample independently per image. Full overnight runs use the
parallel CPU PolicyDataset to avoid serial per-image CUDA launch overhead.
"""
import torch
import torch.nn.functional as F
from torchvision.transforms import RandAugment,AutoAugment,AutoAugmentPolicy


class Augment:
    def __init__(self,stats,device,policy='basic'):
        self.mean=torch.tensor(stats['mean'],device=device).view(1,3,1,1)
        self.std=torch.tensor(stats['std'],device=device).view(1,3,1,1)
        self.policy=policy
        self.advanced=RandAugment(2,9) if policy=='randaugment' else AutoAugment(AutoAugmentPolicy.CIFAR10) if policy=='autoaugment' else None

    def __call__(self,x,train=False,shift=False):
        if train:
            if self.advanced is not None:x=torch.stack([self.advanced(im) for im in x])
            x=x.float()/255
            padded=F.pad(x,(4,4,4,4),mode='reflect');n=len(x)
            offsets=torch.randint(0,9,(n,2),device=x.device)
            r=offsets[:,0,None,None]+torch.arange(32,device=x.device)[None,:,None]
            c=offsets[:,1,None,None]+torch.arange(32,device=x.device)[None,None,:]
            x=padded.permute(0,2,3,1)[torch.arange(n,device=x.device)[:,None,None],r,c].permute(0,3,1,2)
            flip=torch.rand((n,1,1,1),device=x.device)<.5
            x=torch.where(flip,x.flip(-1),x)
            if self.policy=='color':
                gray=(x*torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]).sum(1,keepdim=True)
                sat=torch.rand((n,1,1,1),device=x.device)*.8+.6
                sat=torch.where(torch.rand_like(sat)<.1,torch.zeros_like(sat),sat)
                x=gray+(x-gray)*sat
                contrast=torch.rand((n,1,1,1),device=x.device)*.6+.7
                brightness=torch.rand((n,1,1,1),device=x.device)*.4+.8
                m=x.mean((1,2,3),keepdim=True);x=((x-m)*contrast+m)*brightness;x=x.clamp(0,1)
        else:x=x.float()/255
        if shift:
            # Deliberately mild fixed robustness probe, not a claimed test-shift simulator.
            gray=(x*torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]).sum(1,keepdim=True)
            x=(.5*x+.5*gray).clamp(0,1)
        return ((x-self.mean)/self.std).contiguous(memory_format=torch.channels_last)


def mix_batch(x,y,alpha):
    if alpha<=0:return x,y,y,1.
    lam=float(torch.distributions.Beta(alpha,alpha).sample())
    order=torch.randperm(len(x),device=x.device)
    return x*lam+x[order]*(1-lam),y,y[order],lam
