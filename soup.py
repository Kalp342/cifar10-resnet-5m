"""Uniform model soup (Wortsman et al., ICML 2022): average the weights of several copies of one model
that were each fine-tuned from the same checkpoint with a different seed / learning rate.
The result is a single network with exactly the same architecture and parameter count."""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from train import ResNet, count_params, find_file, find_dir, load_images

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", required=True)
ap.add_argument("--ingredients", required=True, help="comma-separated model_final.pt paths")
ap.add_argument("--base", default="", help="the checkpoint the ingredients were fine-tuned from (for comparison)")
ap.add_argument("--widths", default="64,128,256")
ap.add_argument("--blocks", default="3,5,3")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
W, B = [int(w) for w in args.widths.split(",")], [int(b) for b in args.blocks.split(",")]

def load_sd(p):
    sd = torch.load(p, map_location="cpu")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items() if k != "n_averaged"}

def make(sd):
    m = ResNet(W, B); m.load_state_dict(sd); return m.to(dev).to(memory_format=torch.channels_last).eval()

# ---- data (same normalisation as training on the full 50k set)
train_df = pd.read_csv(find_file(args.data, "train_labels.csv")); test_df = pd.read_csv(find_file(args.data, "test.csv"))
cache = os.path.join(args.data, "decoded_images_cache.npz")
if os.path.exists(cache):
    z = np.load(cache); X_all, X_test = z["X_all"], z["X_test"]
else:
    X_all = load_images(find_dir(args.data, "train_images"), train_df["filename"].tolist())
    X_test = load_images(find_dir(args.data, "test_images"), test_df["filename"].tolist())
y_all = train_df["label"].values
MEAN = torch.tensor(X_all.reshape(-1, 3).mean(0) / 255, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
STD = torch.tensor(X_all.reshape(-1, 3).std(0) / 255, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
to_gpu = lambda X: torch.from_numpy(X).permute(0, 3, 1, 2).contiguous().to(dev)
Xte, Xtr = to_gpu(X_test), to_gpu(X_all[:10000])

@torch.no_grad()
def probs(net, Xg, views=2):
    out = []
    for i in range(0, len(Xg), 500):
        x = (Xg[i:i + 500].float() / 255 - MEAN) / STD
        shifts = [(2, 2)] if views == 2 else [(2, 2), (0, 2), (4, 2), (2, 0), (2, 4)]
        xp = F.pad(x, (2, 2, 2, 2), mode="reflect"); p = 0
        for dy, dx in shifts:
            v = xp[:, :, dy:dy + 32, dx:dx + 32].contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                p = p + net(v).float().softmax(1) + net(v.flip(3)).float().softmax(1)
        out.append(p.cpu())
    return torch.cat(out)

# ---- uniform soup: average every float tensor (weights and BN running stats)
sds = [load_sd(p) for p in args.ingredients.split(",")]
soup = {k: (torch.stack([sd[k].float() for sd in sds]).mean(0).to(sds[0][k].dtype)
            if sds[0][k].is_floating_point() else sds[0][k]) for k in sds[0]}
net = make(soup)
print(f"soup of {len(sds)} ingredients, trainable parameters: {count_params(net):,}")
assert count_params(net) <= 5_000_000

# sanity check: clean accuracy on 10k training images (a broken soup collapses here)
rep = {}
named = [(f"ingredient{i+1}", sd) for i, sd in enumerate(sds)] + ([("base", load_sd(args.base))] if args.base else [])
soup_test = probs(net, Xte, 2).argmax(1).numpy()
for name, sd in named + [("soup", soup)]:
    m = net if name == "soup" else make(sd)
    acc = float((probs(m, Xtr, 2).argmax(1).numpy() == y_all[:10000]).mean())
    agree = float((probs(m, Xte, 2).argmax(1).numpy() == soup_test).mean())
    rep[name] = dict(train10k_acc=acc, test_agreement_with_soup=agree)
    print(f"{name:12s} clean train-10k acc {acc:.4f} | test agreement with soup {agree:.4f}")

p2, p10 = probs(net, Xte, 2).argmax(1).numpy(), probs(net, Xte, 10).argmax(1).numpy()
pd.DataFrame({"id": test_df["id"].values, "label": p2.astype(int)}).to_csv(os.path.join(args.out, "submission.csv"), index=False)
pd.DataFrame({"id": test_df["id"].values, "label": p10.astype(int)}).to_csv(os.path.join(args.out, "submission_tta10.csv"), index=False)
print("prediction distribution:", np.bincount(p10, minlength=10)); print("tta10 changes vs 2-view:", int((p10 != p2).sum()))
torch.save(soup, os.path.join(args.out, "model_final.pt"))
json.dump(rep, open(os.path.join(args.out, "results.json"), "w"), indent=2)
print("ALL_DONE")
