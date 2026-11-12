"""Training script for every model in this repo (NYU Tandon Deep Learning F26, CIFAR-10 under 5M parameters).

One script covers all experiments:
    baseline      python train.py --data data --out baseline --val_size 5000
    teacher       python train.py --data data --out teacher_b11 --val_size 0 --epochs 200 --seed 11 --widths 128,256,512 --teacher
    distillation  python train.py --data data --out kd2_student --val_size 0 --epochs 500 --seed 21 --blocks 3,5,3 \
                      --kd_teachers "teacher_b11/model_final.pt@128,256,512@3,4,3;..." --tta10
    SAM           add --sam_rho 0.05
    soup copies   add --init kd2_student/model_final.pt --epochs 30 --lr 0.02 --warmup_epochs 1 --cutmix_off_epochs 10
    high-res      add --strides 1,1,2
See scripts/ for the exact commands behind each checkpoint.

Model: 3-stage ResNet of BasicBlocks (y = ReLU(F(x) + S(x))), 1x1 projection shortcuts when the shape changes, global
average pooling. Widths 64/128/256 with blocks 3-4-3 = 4,623,178 params; 3-5-3 = 4,918,602 params.
Training: random crop + flip + TrivialAugment-style ops + random erasing (all on GPU), CutMix (off for the last epochs),
label smoothing 0.1, SGD Nesterov + warmup/cosine LR, weight decay 5e-4 (not on BN/bias), mixed precision, EMA of
weights; optional knowledge distillation (KL to the teachers' averaged softened predictions) and SAM.
Inference: EMA weights + flip TTA (submission.csv) and optionally flip x 5 shifts (submission_tta10.csv).
Trained from scratch on the competition's CIFAR-10 training images only.
Resumable: re-running with the same --out continues from the last checkpoint.
"""
import os, sys, glob, time, math, random, json, argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T


# ----------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    """y = ReLU(S(x) + F(x)); F = conv-BN-ReLU-conv-BN, S = identity or 1x1 conv-BN projection."""
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        for m in self.modules():  # zero-init last BN of each residual branch
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------- data
class ArrayDS(Dataset):
    def __init__(self, X, y=None, tf=None):
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
        self.y = None if y is None else torch.from_numpy(y)
        self.tf = tf
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.tf(self.X[i])
        return (x, self.y[i]) if self.y is not None else x


def find_file(root, name):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    assert hits, f"could not find {name} under {root}"
    return hits[0]


def find_dir(root, name):
    hits = [p for p in glob.glob(os.path.join(root, "**", name), recursive=True) if os.path.isdir(p)]
    assert hits, f"could not find folder {name} under {root}"
    hits.sort(key=len, reverse=True)  # deepest match handles train_images/train_images nesting
    return hits[0]


def load_images(folder, files):
    arr = np.empty((len(files), 32, 32, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        arr[i] = np.asarray(Image.open(os.path.join(folder, f)).convert("RGB"))
    return arr


def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[2:]
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
    x1, x2 = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, y, y[idx], lam


# ----------------------------------------------------------------------------- GPU augmentation
# All augmentation runs on the GPU on whole batches (much faster than per-image CPU transforms,
# especially on Windows). Pipeline: random crop (reflect pad 4) -> horizontal flip ->
# TrivialAugmentWide-style op (one random op per image, random magnitude) -> normalise ->
# random erasing. CutMix is applied afterwards in the training loop.
def gpu_crop_flip(x, pad=4):
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    iy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    ix = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    ar = torch.arange(H, device=x.device)
    rows = (iy[:, None] + ar)[:, None, :, None].expand(B, C, H, W)
    cols = (ix[:, None] + ar)[:, None, None, :].expand(B, C, H, W)
    out = xp[torch.arange(B, device=x.device)[:, None, None, None],
             torch.arange(C, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def gpu_trivial_augment(x):
    """x in [0,1], shape (B,3,H,W). Each image gets one op chosen uniformly with a uniform random magnitude."""
    B, C, H, W = x.shape
    dev = x.device
    op = torch.randint(0, 13, (B,), device=dev)
    m = torch.rand(B, device=dev) * 2 - 1                      # signed magnitude in [-1, 1]
    v = lambda t: t[:, None, None, None]
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    out = x.clone()
    # colour ops
    sel = v(op == 1); out = torch.where(sel, x * v(1 + 0.99 * m), out)                                  # brightness
    sel = v(op == 2); out = torch.where(sel, gray + (x - gray) * v(1 + 0.99 * m), out)                    # saturation
    mean = gray.mean(dim=(2, 3), keepdim=True)
    sel = v(op == 3); out = torch.where(sel, mean + (x - mean) * v(1 + 0.99 * m), out)                    # contrast
    bits = (2 + (m.abs() * 6).round()).clamp(2, 8)                                                          # posterize
    q = v(2 ** (8 - bits)); sel = v(op == 4); out = torch.where(sel, torch.floor(x * 255 / q) * q / 255, out)
    thr = v(m.abs()); sel = v(op == 5); out = torch.where(sel, torch.where(x >= thr, 1 - x, x), out)         # solarize
    lo = x.amin(dim=(2, 3), keepdim=True); hi = x.amax(dim=(2, 3), keepdim=True)                         # autocontrast
    sel = v(op == 6); out = torch.where(sel, (x - lo) / (hi - lo).clamp_min(1e-3), out)
    blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)                                  # sharpness
    sel = v(op == 7); out = torch.where(sel, blur + (x - blur) * v(1 + 0.99 * m), out)
    # geometric ops via one affine warp (identity for non-geometric images); fill = 0
    theta = torch.zeros(B, 2, 3, device=dev); theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    ang = m * math.radians(135)
    rot = op == 8
    theta[rot, 0, 0] = torch.cos(ang[rot]); theta[rot, 0, 1] = -torch.sin(ang[rot])
    theta[rot, 1, 0] = torch.sin(ang[rot]); theta[rot, 1, 1] = torch.cos(ang[rot])
    theta[op == 9, 0, 1] = 0.99 * m[op == 9]                                                              # shear x
    theta[op == 10, 1, 0] = 0.99 * m[op == 10]                                                            # shear y
    theta[op == 11, 0, 2] = 0.6 * m[op == 11]                                                             # translate x (<= ~10px)
    theta[op == 12, 1, 2] = 0.6 * m[op == 12]                                                             # translate y
    geo = op >= 8
    if geo.any():
        grid = F.affine_grid(theta[geo], (int(geo.sum()), C, H, W), align_corners=False)
        out[geo] = F.grid_sample(x[geo], grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.clamp(0, 1)


def gpu_random_erasing(x, p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
    B, C, H, W = x.shape
    dev = x.device
    area = torch.empty(B, device=dev).uniform_(*scale) * H * W
    logr = torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]))
    h = (area * logr.exp()).sqrt().round().clamp(1, H).long()
    w = (area / logr.exp()).sqrt().round().clamp(1, W).long()
    y0 = (torch.rand(B, device=dev) * (H - h + 1).float()).long()
    x0 = (torch.rand(B, device=dev) * (W - w + 1).float()).long()
    ar = torch.arange(H, device=dev)
    my = (ar[None] >= y0[:, None]) & (ar[None] < (y0 + h)[:, None])
    mx = (ar[None] >= x0[:, None]) & (ar[None] < (x0 + w)[:, None])
    mask = (my[:, :, None] & mx[:, None, :]) & (torch.rand(B, device=dev) < p)[:, None, None]
    return x.masked_fill(mask[:, None], 0.0)


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()


def keep_awake(on=True):
    """Ask Windows not to sleep while this process runs (reverts automatically when it exits)."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="run_val")
    ap.add_argument("--val_size", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--widths", default="64,128,256", help="channel widths per stage (teachers may be wider)")
    ap.add_argument("--blocks", default="3,4,3")
    ap.add_argument("--teacher", action="store_true", help="teacher model: allowed to exceed 5M params (not submitted)")
    ap.add_argument("--kd_teachers", default="", help="knowledge distillation: ';'-separated list of path/to/model_final.pt@w1,w2,w3")
    ap.add_argument("--kd_alpha", type=float, default=0.9, help="weight of the distillation loss")
    ap.add_argument("--kd_T", type=float, default=4.0, help="distillation temperature")
    ap.add_argument("--tta10", action="store_true", help="also write submission_tta10.csv (flip x 5 shifts)")
    ap.add_argument("--strides", default="1,2,2", help="first-block stride of each stage (1,1,2 keeps 32x32 in stage 2)")
    ap.add_argument("--init", default="", help="start from these weights (model soup ingredient fine-tuning)")
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--cutmix_off_epochs", type=int, default=15)
    ap.add_argument("--sam_rho", type=float, default=0.0,
                    help="Sharpness-Aware Minimization radius (Foret et al. 2021); 0 = plain SGD")
    args = ap.parse_args()
    CFG = dict(vars(args), warmup_epochs=5, weight_decay=5e-4, label_smoothing=0.1, cutmix_prob=0.5,
               cutmix_off_epochs=15, ema_decay=0.999,
               widths=[int(w) for w in args.widths.split(",")], blocks=[int(b) for b in args.blocks.split(",")])

    os.makedirs(args.out, exist_ok=True)
    sys.stdout = Tee(sys.__stdout__, open(os.path.join(args.out, "log.txt"), "a", encoding="utf-8"))
    keep_awake(True)
    print("=" * 80, "\n", time.strftime("%Y-%m-%d %H:%M:%S"), CFG)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", "| torch", torch.__version__)
    torch.backends.cudnn.benchmark = True

    CFG["warmup_epochs"], CFG["cutmix_off_epochs"] = args.warmup_epochs, args.cutmix_off_epochs
    CFG["strides"] = [int(x) for x in args.strides.split(",")]
    model = ResNet(CFG["widths"], CFG["blocks"], strides=CFG["strides"])
    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items() if k != "n_averaged"})
        print("initialised from", os.path.basename(os.path.dirname(args.init)))
    n_params = count_params(model)
    print(f"Trainable parameters: {n_params:,}" + ("  (teacher - not a submission model)" if args.teacher else ""))
    assert args.teacher or n_params <= 5_000_000

    # ---- data
    train_df = pd.read_csv(find_file(args.data, "train_labels.csv"))
    test_df = pd.read_csv(find_file(args.data, "test.csv"))
    y_all = train_df["label"].values.astype(np.int64)
    t0 = time.time()
    cache = os.path.join(args.data, "decoded_images_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache); X_all, X_test = z["X_all"], z["X_test"]
    else:
        train_dir, test_dir = find_dir(args.data, "train_images"), find_dir(args.data, "test_images")
        X_all = load_images(train_dir, train_df["filename"].tolist())
        X_test = load_images(test_dir, test_df["filename"].tolist())
        np.savez(cache, X_all=X_all, X_test=X_test)   # decoded once; later runs load in seconds
    print(f"loaded {len(X_all)} train / {len(X_test)} test images in {time.time()-t0:.0f}s; class counts {np.bincount(y_all)}")

    rng = np.random.RandomState(args.seed)
    if args.val_size > 0:
        per_class = args.val_size // 10
        val_idx = np.concatenate([rng.permutation(np.where(y_all == c)[0])[:per_class] for c in range(10)])
        mask = np.ones(len(y_all), bool); mask[val_idx] = False
        X_tr, y_tr, X_val, y_val = X_all[mask], y_all[mask], X_all[val_idx], y_all[val_idx]
    else:
        X_tr, y_tr, X_val, y_val = X_all, y_all, None, None
    print("train", X_tr.shape, "val", None if X_val is None else X_val.shape)

    MEAN = torch.tensor(X_tr.reshape(-1, 3).mean(0) / 255, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    STD = torch.tensor(X_tr.reshape(-1, 3).std(0) / 255, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    to_gpu = lambda X: torch.from_numpy(X).permute(0, 3, 1, 2).contiguous().to(device)   # uint8 NCHW on GPU
    Xtr_g, ytr_g = to_gpu(X_tr), torch.from_numpy(y_tr).to(device)
    Xval_g = None if X_val is None else to_gpu(X_val)
    Xte_g = to_gpu(X_test)

    def train_batches():
        perm = torch.randperm(len(Xtr_g), device=device)
        for i in range(len(perm) // args.batch_size):        # drop_last
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            x = Xtr_g[idx].float() / 255
            x = gpu_trivial_augment(gpu_crop_flip(x))
            x = gpu_random_erasing((x - MEAN) / STD, p=0.25)
            yield x, ytr_g[idx]

    @torch.no_grad()
    def predict_probs(net, Xg, tta=True):
        net.eval(); out = []
        for i in range(0, len(Xg), 1000):
            x = ((Xg[i:i + 1000].float() / 255 - MEAN) / STD).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                p = net(x).float().softmax(1)
                if tta:
                    p = p + net(torch.flip(x, dims=[3])).float().softmax(1)
            out.append(p.cpu())
        return torch.cat(out)

    def accuracy(net, Xg, labels, tta=True):
        return float((predict_probs(net, Xg, tta).argmax(1).numpy() == labels).mean())
    val_dl, test_dl = Xval_g, Xte_g

    # ---- optimisation
    model = model.to(device).to(memory_format=torch.channels_last)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(CFG["ema_decay"]), use_buffers=True)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": CFG["weight_decay"]},
                           {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, momentum=0.9, nesterov=True)
    steps_per_epoch = len(X_tr) // args.batch_size
    total_steps, warmup_steps = args.epochs * steps_per_epoch, CFG["warmup_epochs"] * steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

    # ---- knowledge distillation teachers (trained from scratch on the same training images)
    teachers = []
    for spec in [t for t in args.kd_teachers.split(";") if t.strip()]:
        parts = spec.split("@")
        path, widths = parts[0], parts[1]
        tblocks = [int(b) for b in parts[2].split(",")] if len(parts) > 2 else [3, 4, 3]
        tnet = ResNet([int(w) for w in widths.split(",")], tblocks).to(device).to(memory_format=torch.channels_last)
        sd = torch.load(path, map_location=device)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items() if k != "n_averaged"}
        tnet.load_state_dict(sd); tnet.eval()
        for p_ in tnet.parameters(): p_.requires_grad_(False)
        teachers.append(tnet)
        print(f"KD teacher {path} widths={widths} params={sum(p.numel() for p in tnet.parameters()):,}")
    T_kd = args.kd_T

    def teacher_probs(x):
        with torch.no_grad():
            return torch.stack([t(x).float().div(T_kd).softmax(1) for t in teachers]).mean(0)

    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    start_epoch, history = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, history = ck["epoch"], ck["history"]
        print(f"resumed from checkpoint at epoch {start_epoch}")

    bn_layers = [m_ for m_ in model.modules() if isinstance(m_, nn.BatchNorm2d)]
    bn_moms = [m_.momentum for m_ in bn_layers]
    if args.sam_rho > 0: print(f"SAM enabled, rho={args.sam_rho}")
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        use_cutmix = epoch < args.epochs - CFG["cutmix_off_epochs"]
        tot_loss, tot_n, t0 = 0.0, 0, time.time()
        for x, y in train_batches():
            x = x.contiguous(memory_format=torch.channels_last)
            if use_cutmix and np.random.rand() < CFG["cutmix_prob"]:
                x, ya, yb, lam = cutmix(x, y)
            else:
                ya, yb, lam = y, y, 1.0
            pt = None
            if teachers:   # teacher ensemble's softened predictions on the same augmented batch
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                    pt = teacher_probs(x)

            def batch_loss():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                    out = model(x)
                    l = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb) if lam < 1.0 else criterion(out, y)
                    if pt is not None:
                        kd = F.kl_div(F.log_softmax(out.float() / T_kd, 1), pt, reduction="batchmean") * T_kd * T_kd
                        l = args.kd_alpha * kd + (1 - args.kd_alpha) * l
                return l

            opt.zero_grad(set_to_none=True)
            loss = batch_loss()
            scaler.scale(loss).backward()
            if args.sam_rho > 0:
                # SAM: move to the worst-case point w + rho*g/||g|| and take the gradient there
                s_ = scaler.get_scale()
                params = [p_ for p_ in model.parameters() if p_.grad is not None]
                gnorm = torch.norm(torch.stack([(p_.grad.float() / s_).norm() for p_ in params]))
                if torch.isfinite(gnorm):
                    with torch.no_grad():
                        eps = [(p_.grad.float() / s_) * (args.sam_rho / (gnorm + 1e-12)) for p_ in params]
                        for p_, e_ in zip(params, eps): p_.add_(e_.to(p_.dtype))
                    opt.zero_grad(set_to_none=True)
                    for m_ in bn_layers: m_.momentum = 0.0          # don't update BN stats twice
                    scaler.scale(batch_loss()).backward()
                    for m_, mom in zip(bn_layers, bn_moms): m_.momentum = mom
                    with torch.no_grad():
                        for p_, e_ in zip(params, eps): p_.sub_(e_.to(p_.dtype))
            scaler.step(opt); scaler.update(); sched.step()
            ema.update_parameters(model)
            tot_loss += loss.item() * x.size(0); tot_n += x.size(0)
        rec = dict(epoch=epoch + 1, train_loss=tot_loss / tot_n, lr=opt.param_groups[0]["lr"], time=time.time() - t0)
        if val_dl is not None and ((epoch + 1) % 5 == 0 or epoch + 1 == args.epochs or epoch < 2):
            rec["val_acc"] = accuracy(model, val_dl, y_val, tta=False)
            rec["val_acc_ema"] = accuracy(ema.module, val_dl, y_val, tta=False)
        history.append(rec)
        eta = (args.epochs - epoch - 1) * rec["time"] / 60
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()) + f" eta={eta:.0f}min")
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), scaler=scaler.state_dict(), epoch=epoch + 1,
                            history=history), ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    print(f"training time this session {(time.time()-t_start)/60:.1f} min")
    pd.DataFrame(history).to_csv(os.path.join(args.out, "training_log.csv"), index=False)

    # ---- evaluation & submission
    final, results = ema.module, {"params": n_params}
    if val_dl is not None:
        for name, net in [("model", model), ("ema", ema.module)]:
            results[f"{name}_val_acc"] = accuracy(net, val_dl, y_val, tta=False)
            results[f"{name}_val_acc_tta"] = accuracy(net, val_dl, y_val, tta=True)
        print("validation:", results)
        if results["model_val_acc_tta"] > results["ema_val_acc_tta"]:
            final = model
    preds = predict_probs(final, test_dl, tta=True).argmax(1).numpy()
    sub = pd.DataFrame({"id": test_df["id"].values, "label": preds.astype(int)})
    sub.to_csv(os.path.join(args.out, "submission.csv"), index=False)
    print("prediction distribution:", np.bincount(preds, minlength=10))
    if args.tta10:   # 10-view test-time augmentation: {identity, flip} x {0, +-2px shifts in x and y}
        final.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(Xte_g), 500):
                x = ((Xte_g[i:i + 500].float() / 255 - MEAN) / STD)
                xp = F.pad(x, (2, 2, 2, 2), mode="reflect"); p = 0
                for dy, dx in [(2, 2), (0, 2), (4, 2), (2, 0), (2, 4)]:
                    v = xp[:, :, dy:dy + 32, dx:dx + 32].contiguous(memory_format=torch.channels_last)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                        p = p + final(v).float().softmax(1) + final(v.flip(3)).float().softmax(1)
                outs.append(p.cpu())
        preds10 = torch.cat(outs).argmax(1).numpy()
        pd.DataFrame({"id": test_df["id"].values, "label": preds10.astype(int)}).to_csv(
            os.path.join(args.out, "submission_tta10.csv"), index=False)
        print("tta10 changes vs 2-view:", int((preds10 != preds).sum()))
    torch.save(final.state_dict(), os.path.join(args.out, "model_final.pt"))
    json.dump({**results, "config": CFG}, open(os.path.join(args.out, "results.json"), "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        h = pd.DataFrame(history)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(h.epoch, h.train_loss); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
        if "val_acc" in h:
            v = h.dropna(subset=["val_acc"])
            ax[1].plot(v.epoch, v.val_acc, label="model"); ax[1].plot(v.epoch, v.val_acc_ema, label="EMA")
            ax[1].set_title("validation accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "curves.png"), dpi=120)
    except Exception as e:
        print("plot skipped:", e)
    keep_awake(False)
    print("ALL_DONE")


if __name__ == "__main__":
    main()
