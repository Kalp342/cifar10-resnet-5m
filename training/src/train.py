"""Single-GPU trainer: resident uint8 data, AMP, EMA, atomic resumable checkpoints."""
import argparse,copy,csv,json,math,os,random,time,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .models import build_model,enforce_budget
from .dataset import ROOT,load_data
from .augmentations import Augment,mix_batch


def atomic_save(obj,path):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);os.replace(temp,path)


def setup(seed,deterministic=False):
    if deterministic:os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=not deterministic
    torch.use_deterministic_algorithms(deterministic)
    torch.set_float32_matmul_precision('high')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.inference_mode()
def evaluate(model,images,labels,ids,aug,batch=256,tta=False,shift=False,amp=True):
    model.eval();loss=torch.zeros((),device=images.device);correct=torch.zeros_like(loss);pred=[]
    for ix in ids.split(batch):
        x=aug(images[ix],shift=shift)
        with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=amp and images.device.type=='cuda'):
            logits=model(x)
            if tta:
                p=(logits.float().softmax(1)+model(x.flip(-1)).float().softmax(1))*.5
                batch_loss=F.nll_loss(p.clamp_min(1e-8).log(),labels[ix],reduction='sum')
            else:p=logits.float().softmax(1);batch_loss=F.cross_entropy(logits.float(),labels[ix],reduction='sum')
        loss+=batch_loss;correct+=(p.argmax(1)==labels[ix]).sum();pred.append(p.cpu())
    return {'loss':loss.item()/len(ids),'accuracy':correct.item()/len(ids)},torch.cat(pred)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'].cpu())
    if r['cuda'] is not None:torch.cuda.set_rng_state_all([s.cpu() for s in r['cuda']])


@torch.inference_mode()
def recalibrate_bn(model,images,ids,aug,amp,batch=256,limit=4096):
    """EMA weights need their own BN statistics; never use validation/test samples."""
    model.eval();layers=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    momenta=[m.momentum for m in layers]
    for layer in layers:layer.reset_running_stats();layer.momentum=None;layer.train()
    for ix in ids[:limit].split(batch):
        with torch.autocast(images.device.type,dtype=torch.bfloat16,enabled=amp):model(aug(images[ix]))
    for layer,momentum in zip(layers,momenta):layer.momentum=momentum;layer.eval()
    model.eval()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--epochs',type=int);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--full-data',action='store_true');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());cfg['full_data']=args.full_data or cfg.get('full_data',False)
    if args.epochs is not None:cfg['epochs']=args.epochs
    device=setup(cfg.get('seed',42),cfg.get('deterministic',False))
    amp=cfg.get('amp',True) and device.type=='cuda'
    if amp and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 is required by this config; disable amp for this device')
    run=ROOT/'outputs'/'checkpoints'/cfg['name'];run.mkdir(parents=True,exist_ok=True)
    if (run/'last.pt').exists() and not args.resume:raise FileExistsError(f'{run}: use --resume or a new run name')
    if args.max_steps and cfg['name']!='smoke':raise ValueError('--max-steps is restricted to the smoke config')
    images,labels,train_ids,val_ids,stats=load_data(device,cfg['full_data']);aug=Augment(stats,device,cfg.get('augmentation','basic'))
    model=build_model(cfg).to(device).to(memory_format=torch.channels_last)
    counts=enforce_budget(model)
    ema=copy.deepcopy(model).eval() if cfg.get('ema',True) else None
    if ema:
        for param in ema.parameters():param.requires_grad_(False)
    if cfg.get('optimizer','sgd')=='sgd':
        opt=torch.optim.SGD(model.parameters(),lr=cfg['lr'],momentum=.9,nesterov=True,weight_decay=cfg.get('weight_decay',5e-4),foreach=True)
    elif cfg['optimizer']=='adamw':
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg.get('weight_decay',5e-4),foreach=True)
    else:raise ValueError(f'Unknown optimizer {cfg["optimizer"]}')
    start=0;best=-1.;elapsed_before=0.;steps=0
    if args.resume:
        checkpoint=torch.load(run/'last.pt',map_location=device,weights_only=False)
        # The schedule horizon, split, optimizer and augmentation must stay identical on resume.
        if checkpoint['config']!=cfg:raise ValueError('Resume requires identical config; create a new run for changed experiments')
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        if ema:ema.load_state_dict(checkpoint['ema'])
        start=checkpoint['epoch']+1;best=checkpoint['best'];elapsed_before=checkpoint['elapsed_seconds'];steps=checkpoint['steps']
        restore_rng(checkpoint['rng'])
        # A crash between log append and atomic checkpoint replace can leave an extra row.
        if (run/'history.csv').exists():
            with (run/'history.csv').open(newline='') as f:
                reader=csv.DictReader(f);fields=reader.fieldnames;rows=[r for r in reader if int(r['epoch'])<=start]
            with (run/'history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    elif cfg.get('init_checkpoint'):
        initial=torch.load(ROOT/cfg['init_checkpoint'],map_location=device,weights_only=False)
        assert initial['parameters']==counts
        assert initial['manifest']==json.loads((ROOT/'outputs/manifest.json').read_text())
        # Fine-tuning branches restart optimizer and EMA from identical parent weights.
        weights=initial[cfg.get('init_weights','model')]
        model.load_state_dict(weights)
        if ema:ema.load_state_dict(weights)
    (run/'config.json').write_text(json.dumps(cfg,indent=2))
    if start>=cfg['epochs']:
        print(f'{cfg["name"]} already completed {start} epochs.',flush=True);return
    diagnostic={'torch':torch.__version__,'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None,
        'vram_gib':torch.cuda.get_device_properties(0).total_memory/2**30 if device.type=='cuda' else 0,
        'parameters':counts,'train_rows':len(train_ids),'validation_rows':0 if cfg['full_data'] else len(val_ids),'amp':amp,
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'src').glob('*.py'))}}
    (run/'environment.json').write_text(json.dumps(diagnostic,indent=2));print(json.dumps(diagnostic),flush=True)
    snapshot=run/'source';snapshot.mkdir(exist_ok=True)
    for source in (ROOT/'src').glob('*.py'):
        target=snapshot/source.name
        if not target.exists():target.write_bytes(source.read_bytes())
    print(f'Run {cfg["name"]}, epochs {start+1}..{cfg["epochs"]}',flush=True)
    started=time.perf_counter();batch=cfg.get('batch_size',128);total_steps=math.ceil(len(train_ids)/batch)
    warmup=cfg.get('warmup_epochs',5);max_epochs=cfg['epochs'];logpath=run/'history.csv'
    for epoch in range(start,max_epochs):
        epoch_started=time.perf_counter();model.train();order=train_ids[torch.randperm(len(train_ids),device=device)]
        train_loss=torch.zeros((),device=device);train_correct=torch.zeros_like(train_loss);seen=0
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for b,ix in enumerate(order.split(batch)):
            progress=epoch+b/total_steps
            multiplier=(progress+1/total_steps)/warmup if progress<warmup else .5*(1+math.cos(math.pi*(progress-warmup)/max(1,max_epochs-warmup)))
            lr=cfg['lr']*max(multiplier,.001)
            for group in opt.param_groups:group['lr']=lr
            x=aug(images[ix],train=True);y=labels[ix]
            x,ya,yb,lam=mix_batch(x,y,cfg.get('mixup',0))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
                logits=model(x);smoothing=cfg.get('label_smoothing',0)
                loss=lam*F.cross_entropy(logits,ya,label_smoothing=smoothing)+(1-lam)*F.cross_entropy(logits,yb,label_smoothing=smoothing)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}, batch {b}')
            loss.backward()
            if cfg.get('grad_clip',0)>0:torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
            opt.step();steps+=1
            if ema:
                decay=min(cfg.get('ema_decay',.999),(1+steps)/(10+steps))
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),1-decay)
            train_loss+=loss.detach()*len(ix)
            predicted=logits.argmax(1)
            if cfg.get('mixup',0)>0:
                train_correct+=lam*(predicted==ya).sum()+(1-lam)*(predicted==yb).sum()
            else:train_correct+=(predicted==y).sum()
            seen+=len(ix)
            if args.max_steps and b+1>=args.max_steps:break
        if ema:recalibrate_bn(ema,images,train_ids,aug,amp,limit=cfg.get('ema_bn_samples',4096))
        # A full-data refit must never present training examples as held-out validation.
        val={'loss':float('nan'),'accuracy':float('nan')};ev=dict(val);chosen='model';score=-1.
        if not cfg['full_data']:
            val,_=evaluate(model,images,labels,val_ids,aug,amp=amp)
            if ema:ev,_=evaluate(ema,images,labels,val_ids,aug,amp=amp)
            score=val['accuracy']
            if ema and ev['accuracy']>score:score=ev['accuracy'];chosen='ema'
        improved=score>best or cfg['full_data'];best=max(best,score)
        elapsed=elapsed_before+time.perf_counter()-started
        row={'epoch':epoch+1,'lr':lr,'train_loss':train_loss.item()/seen,'train_accuracy':train_correct.item()/seen,
            'val_loss':val['loss'],'val_accuracy':val['accuracy'],'ema_loss':ev['loss'],'ema_accuracy':ev['accuracy'],
            'epoch_seconds':time.perf_counter()-epoch_started,'elapsed_seconds':elapsed,
            'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0,'images_per_second':seen/(time.perf_counter()-epoch_started)}
        with logpath.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=row)
            if f.tell()==0:writer.writeheader()
            writer.writerow(row)
        state={'model':model.state_dict(),'ema':ema.state_dict() if ema else None,'optimizer':opt.state_dict(),'config':cfg,
            'epoch':epoch,'steps':steps,'best':best,'selected':chosen,'elapsed_seconds':elapsed,'rng':rng_state(),'stats':stats,
            'manifest':json.loads((ROOT/'outputs/manifest.json').read_text()),'parameters':counts}
        atomic_save(state,run/'last.pt')
        if improved:atomic_save(state,run/'best.pt')
        print(json.dumps(row),flush=True)
        # This file provides an orderly, resumable stop at an epoch boundary.
        if (run/'STOP').exists():print('STOP file detected; checkpoint saved. Remove STOP before resuming.',flush=True);break
    summary={'name':cfg['name'],'parameters':counts,'best_validation_accuracy':None if cfg['full_data'] else best,
        'last_epoch':epoch+1,'elapsed_seconds':elapsed,'complete':epoch+1==max_epochs,'full_data':cfg['full_data']}
    (run/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
