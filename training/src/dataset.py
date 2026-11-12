"""Strict metadata mapping, audited pixel caches, fixed leakage-aware split."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import pandas as pd
import torch

ROOT=Path(os.environ.get('PROJECT1_ROOT',Path(__file__).resolve().parents[1]))


def load_data(device='cpu',full=False):
    stats=json.loads((ROOT/'outputs/eda/statistics.json').read_text())
    manifest=json.loads((ROOT/'outputs/manifest.json').read_text())
    frame=pd.read_csv(ROOT/'train_labels.csv')
    assert hashlib.sha256((ROOT/'train_labels.csv').read_bytes()).hexdigest()==manifest['train_labels_sha256']
    x=np.load(ROOT/'outputs/train_images.npy')
    assert x.shape==(len(frame),32,32,3) and x.dtype==np.uint8
    assert hashlib.sha256(x.tobytes()).hexdigest()==manifest['train_pixels_sha256']
    split=np.load(ROOT/'outputs/split_seed42.npz')
    assert not set(split['groups'][split['train']]) & set(split['groups'][split['val']])
    images=torch.from_numpy(x).permute(0,3,1,2).contiguous(memory_format=torch.channels_last).to(device)
    labels=torch.tensor(frame.label.to_numpy(),device=device,dtype=torch.long)
    ti=np.arange(len(frame)) if full else split['train']
    return images,labels,torch.tensor(ti,device=device),torch.tensor(split['val'],device=device),stats


def test_data(device):
    manifest=json.loads((ROOT/'outputs/manifest.json').read_text())
    assert hashlib.sha256((ROOT/'test.csv').read_bytes()).hexdigest()==manifest['test_csv_sha256']
    frame=pd.read_csv(ROOT/'test.csv');x=np.load(ROOT/'outputs/test_images.npy')
    assert hashlib.sha256(x.tobytes()).hexdigest()==manifest['test_pixels_sha256']
    assert len(x)==len(frame) and frame.id.is_unique
    return torch.from_numpy(x).permute(0,3,1,2).contiguous(memory_format=torch.channels_last).to(device),frame
