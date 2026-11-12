from pathlib import Path
import os,json,hashlib
import numpy as np,torch
from src.overnight_common import write_json,utcnow

ROOT=Path(os.environ.get('PROJECT1_ROOT',Path(__file__).resolve().parents[1]))
OUT=ROOT/'outputs/next_stage'
SPECS={
 'A':dict(checkpoint='outputs/overnight/experiments/exp_a_autoaugment/best.pt',weights='model',epoch=283,parameters=4480474,accuracy=.969,reference='outputs/overnight/experiments/exp_a_autoaugment'),
 'B':dict(checkpoint='outputs/overnight/experiments/exp_b_randaugment_mixup/best.pt',weights='ema',epoch=290,parameters=4480474,accuracy=.9678,reference='outputs/overnight/experiments/exp_b_randaugment_mixup'),
 'C':dict(checkpoint='outputs/overnight/experiments/exp_c_se_resnet164/latest.pt',weights='model',epoch=300,parameters=4260214,accuracy=.9686,reference='outputs/overnight/morning_review/C_latest')}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def source_hash():
    base=Path(__file__).resolve().parents[1]
    files=[*sorted((base/'src').glob('*.py')),*sorted((base/'phase4').glob('*.py'))]
    return hashlib.sha256(b''.join(p.relative_to(base).as_posix().encode()+p.read_bytes() for p in files)).hexdigest()

def protected_files():
    files=[ROOT/s['checkpoint'] for s in SPECS.values()]+list((ROOT/'src').glob('*.py'))
    files += [ROOT/p for p in ['project1.pdf','train_labels.csv','test.csv','sample_submission.csv','outputs/split_seed42.npz','outputs/manifest.json','outputs/train_images.npy','outputs/test_images.npy']]
    files += list((ROOT/'outputs/submissions').glob('*.*'))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files if p.is_file()}

def verify_protected():
    original=read(OUT/'audit.json')['protected']
    for name,digest in original.items():assert sha(ROOT/name)==digest,'Protected artifact changed: '+name

def metrics(prob,y,reference=None):
    prob=np.asarray(prob);pred=prob.argmax(1);correct=pred==y
    result={'n':len(y),'accuracy':float(correct.mean()),'nll':float(-np.log(prob[np.arange(len(y)),y].clip(1e-12)).mean()),
            'per_class_accuracy':[float(correct[y==c].mean()) for c in range(10)]}
    if reference is not None:
        old=np.asarray(reference).argmax(1);was=old==y;hard=np.asarray(reference).max(1)<=np.quantile(np.asarray(reference).max(1),.2)
        result.update(changed=int((pred!=old).sum()),corrected=int((correct&~was).sum()),introduced=int((~correct&was).sum()),hard_accuracy=float(correct[hard].mean()))
    return result

@torch.inference_mode()
def views(model,images,normalize,batch=256):
    model.eval();a=[];b=[]
    for pixels in images.split(batch):
        x=normalize(pixels)
        with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=images.device.type=='cuda'):
            a.append(model(x).float().cpu());b.append(model(x.flip(-1)).float().cpu())
    a=torch.cat(a);b=torch.cat(b);p=(a.softmax(1)+b.softmax(1))*.5
    return a.numpy(),b.numpy(),p.numpy()

def load_teacher(name,device):
    from src.models import build_model,enforce_budget
    from src.augmentations import Augment
    s=SPECS[name];cp=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
    assert cp['epoch']+1==s['epoch'] and cp['parameters']['total']==s['parameters']
    model=build_model(cp['config']).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(cp[s['weights']]);model.eval();assert enforce_budget(model)['total']==s['parameters']
    for p in model.parameters():p.requires_grad_(False)
    return model,Augment(cp['stats'],device),cp
