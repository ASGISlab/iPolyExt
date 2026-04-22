# -*- coding: utf-8 -*-
"""
Train legend/patch classifier (timm backbone) with:
- scan train/val from folders (label = first-level folder)
- label_to_idx per run + save pkl (main + timestamp backup)
- train tf: RandomCrop; val tf: CenterCrop
- hierarchical group-balanced sampler: class balanced, and within each class domain balanced
- warmup + cosine LR
- logging to file + stdout
- checkpoints every N epochs
"""

import os
import sys
import math
import time
import json
import yaml
import pickle
import random
import datetime
import logging
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm
from torchvision import transforms
from sklearn.metrics import f1_score


# -----------------------
# utils
# -----------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def setup_logger(log_dir: Path) -> logging.Logger:
    ensure_dir(log_dir)
    log_path = log_dir / f"train_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger("train_legend_classifier")
    logger.info("log_path = %s", str(log_path))
    return logger

def set_seed(seed: int, deterministic: bool, benchmark: bool, logger: logging.Logger):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)

    logger.info("SEED=%d, deterministic=%s, benchmark=%s", seed, deterministic, benchmark)

def maybe_enable_tf32(enable: bool, logger: logging.Logger):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(enable)
        torch.backends.cudnn.allow_tf32 = bool(enable)
        logger.info("TF32=%s", enable)

def resolve_paths(cfg: dict) -> dict:
    base_dir = Path(cfg["paths"]["base_dir"]).expanduser()
    train_dir = cfg["paths"].get("train_dir") or str(base_dir / "train")
    val_dir   = cfg["paths"].get("val_dir")   or str(base_dir / "val")

    out_dir = Path(cfg["paths"]["output_dir"]).expanduser()

    cfg["paths"]["train_dir"] = str(Path(train_dir).expanduser())
    cfg["paths"]["val_dir"]   = str(Path(val_dir).expanduser())
    cfg["paths"]["output_dir"] = str(out_dir)

    return cfg


# -----------------------
# data
# -----------------------
def scan_split(root_dir: str, img_ext: str):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"找不到目錄：{root}")

    img_paths, labels = [], []
    for p in sorted(root.rglob(f"*{img_ext}")):
        rel = p.relative_to(root)
        if len(rel.parts) < 2:
            continue
        cls = rel.parts[0]
        img_paths.append(str(p))
        labels.append(cls)
    return img_paths, labels

def build_label_mapping(train_lbl, val_lbl, logger: logging.Logger):
    train_classes = sorted(set(train_lbl))
    val_classes   = sorted(set(val_lbl))

    unknown = sorted(set(val_classes) - set(train_classes))
    if unknown:
        raise RuntimeError(f"Val 存在未出現在 Train 的類別：{unknown}")

    label_to_idx = {c: i for i, c in enumerate(train_classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    logger.info("num_classes=%d", len(train_classes))
    logger.info("label_to_idx sample: %s", list(label_to_idx.items())[:10])
    return train_classes, label_to_idx, idx_to_label

class CustomDataset(Dataset):
    def __init__(self, paths, labs, lab2idx, tfm):
        self.paths = paths
        self.labs = labs
        self.lab2idx = lab2idx
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.tfm:
            img = self.tfm(img)
        y = self.lab2idx[self.labs[i]]
        return img, y

def extract_domain(path_str: str, root_dir: str) -> str:
    """Assume: root/<class>/<domain>/xxx.png ; if missing -> UNK"""
    rel = Path(path_str).relative_to(Path(root_dir))
    return rel.parts[1] if len(rel.parts) >= 2 else "UNK"


def build_hier_sampler(train_img, train_lbl, train_root, logger: logging.Logger, generator=None):
    # collect (class, domain)
    train_pairs = []
    for p, cls in zip(train_img, train_lbl):
        dom = extract_domain(p, train_root)
        train_pairs.append((cls, dom))

    domains_of_class = defaultdict(set)
    count_cd = Counter(train_pairs)
    for (c, d), n in count_cd.items():
        domains_of_class[c].add(d)

    sample_weights = []
    for (c, d) in train_pairs:
        k_c  = max(1, len(domains_of_class[c]))
        n_cd = max(1, count_cd[(c, d)])
        w_i  = 1.0 / (k_c * n_cd)
        sample_weights.append(w_i)

    # optional diagnostics
    sum_class  = defaultdict(float)
    sum_domain = defaultdict(float)
    for (c, d), w in zip(train_pairs, sample_weights):
        sum_class[c] += w
        sum_domain[(c, d)] += w

    logger.info("class total weight (should ~equal): %s",
                sorted(sum_class.items(), key=lambda x: x[0])[:20])
    logger.info("domain total weight (per class ~ 1/|D_c|): %s",
                sorted(sum_domain.items(), key=lambda x: (x[0][0], x[0][1]))[:20])

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_img),
        replacement=True,
        generator=generator,
    )
    return sampler

# -----------------------
# model/optim/sched
# -----------------------
def build_model(cfg: dict, num_classes: int, target_size, logger: logging.Logger):
    name = cfg["model"]["name"]
    pretrained = bool(cfg["model"].get("pretrained", True))

    kwargs = dict(
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if bool(cfg["model"].get("img_size_override", True)):
        kwargs["img_size"] = tuple(target_size)  # (H, W)

    model = timm.create_model(name, **kwargs)
    logger.info("model=%s, pretrained=%s", name, pretrained)
    return model

def build_optimizer(cfg: dict, model: torch.nn.Module):
    opt_cfg = cfg["train"]["optimizer"]
    lr = float(opt_cfg["lr"])
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))
    wd = float(opt_cfg.get("weight_decay", 0.05))

    no_decay_rules = opt_cfg.get("no_decay_on", ["bias", "norm"])

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        n_low = n.lower()
        is_bias = n.endswith(".bias")
        is_norm = ("norm" in n_low)
        is_1d = (p.ndim == 1)

        if is_1d or is_bias or is_norm:
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = optim.AdamW(
        [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr, betas=betas, eps=eps
    )
    return optimizer

def build_scheduler(cfg: dict, optimizer):
    sch_cfg = cfg["train"]["scheduler"]
    name = sch_cfg.get("name", "warmup_cosine")

    if name != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler: {name}")

    num_epochs = int(cfg["train"]["epochs"])
    warmup_epochs = int(cfg["train"]["warmup_epochs"])

    def lr_lambda(step_epoch):
        # scheduler.step() 在 epoch 結束呼叫，影響「下一個 epoch」
        next_epoch = step_epoch + 2  # after epoch1 step -> next_epoch=2
    
        if warmup_epochs > 0 and next_epoch <= warmup_epochs:
            return float(next_epoch) / float(warmup_epochs)  # ✅ 2/5, 3/5, ..., 5/5
    
        progress = (next_epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -----------------------
# train/eval
# -----------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, f1_avg="macro"):
    model.eval()
    val_sum_loss, correct, total = 0.0, 0, 0
    all_preds, all_gts = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        val_sum_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.detach().cpu().numpy())
        all_gts.extend(labels.detach().cpu().numpy())

    val_loss = val_sum_loss / max(1, total)
    val_acc = 100.0 * correct / max(1, total)
    val_f1 = f1_score(all_gts, all_preds, average=f1_avg)
    return val_loss, val_acc, val_f1

def train_one_epoch(model, loader, criterion, optimizer, device, use_amp: bool, scaler):
    model.train()
    train_sum_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_sum_loss += loss.item() * images.size(0)
        total += images.size(0)

    train_loss = train_sum_loss / max(1, total)
    return train_loss

def save_checkpoint(out_dir: Path, epoch: int, model, label_to_idx: dict, logger: logging.Logger):
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    ckpt_path = ckpt_dir / f"model_epoch_{epoch:02}.pth"
    class_names = [k for k, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]

    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "class_names": class_names,
        },
        ckpt_path
    )
    logger.info("Saved checkpoint: %s", str(ckpt_path))


# -----------------------
# main
# -----------------------
def main(yaml_obj=None):
    if yaml_obj is None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg", type=str, required=True, help="path to yaml config")
        args = parser.parse_args()

        cfg = load_yaml(args.cfg)
    else:
        cfg = yaml_obj
    cfg = resolve_paths(cfg)

    out_dir = Path(cfg["paths"]["output_dir"])
    ensure_dir(out_dir)

    logger = setup_logger(out_dir / "logs")
    logger.info("cfg=%s", json.dumps(cfg, ensure_ascii=False, indent=2))

    # seed / tf32
    set_seed(
        seed=int(cfg["train"]["seed"]),
        deterministic=bool(cfg["train"]["deterministic"]),
        benchmark=bool(cfg["train"]["benchmark"]),
        logger=logger,
    )
    maybe_enable_tf32(bool(cfg["train"].get("tf32", True)), logger)

    # paths
    train_dir = cfg["paths"]["train_dir"]
    val_dir   = cfg["paths"]["val_dir"]

    # scan
    img_ext = cfg["data"].get("img_ext", ".png")
    train_img, train_lbl = scan_split(train_dir, img_ext)
    val_img,   val_lbl   = scan_split(val_dir, img_ext)
    logger.info("scan done: train=%d, val=%d, #train_classes=%d",
                len(train_img), len(val_img), len(set(train_lbl)))

    # label mapping
    train_classes, label_to_idx, idx_to_label = build_label_mapping(train_lbl, val_lbl, logger)

    # save mapping (main + timestamp)
    mapping_dir = out_dir / "mappings"
    ensure_dir(mapping_dir)
    pkl_main = mapping_dir / "label_to_idx.pkl"
    pkl_ts   = mapping_dir / f"label_to_idx_{datetime.datetime.now():%Y%m%d_%H%M%S}.pkl"
    with open(pkl_main, "wb") as f:
        pickle.dump(label_to_idx, f)
    with open(pkl_ts, "wb") as f:
        pickle.dump(label_to_idx, f)
    logger.info("Saved label_to_idx: %s (backup: %s)", str(pkl_main), str(pkl_ts))

    # transforms
    target_size = tuple(cfg["data"]["target_size"])  # (H,W)
    mean = cfg["data"]["normalize_mean"]
    std  = cfg["data"]["normalize_std"]

    train_tf = transforms.Compose([
        transforms.RandomCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # datasets
    train_ds = CustomDataset(train_img, train_lbl, label_to_idx, train_tf)
    val_ds   = CustomDataset(val_img,   val_lbl,   label_to_idx, val_tf)

    # sampler
    use_sampler = bool(cfg["sampler"].get("enable", True))
    sampler = None
    
    g_sampler = torch.Generator()
    g_sampler.manual_seed(int(cfg["train"]["seed"]))
    
    if use_sampler:
        sampler = build_hier_sampler(train_img, train_lbl, train_dir, logger, generator=g_sampler)
        shuffle = False
    else:
        shuffle = True

    
    # dataloaders
    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])
    pin = bool(cfg["data"]["pin_memory"])
    prefetch = int(cfg["data"]["prefetch_factor"])
    persist = bool(cfg["data"]["persistent_workers"]) and (nw > 0)

    g_dl = torch.Generator()
    g_dl.manual_seed(int(cfg["train"]["seed"]))


    train_dl_kwargs = dict(
        dataset=train_ds,
        batch_size=bs,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=persist,
        generator=g_dl,
    )
    
    val_dl_kwargs = dict(
        dataset=val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=persist,
    )
    
    if nw > 0:
        train_dl_kwargs["prefetch_factor"] = prefetch
        val_dl_kwargs["prefetch_factor"] = prefetch
    
    train_loader = DataLoader(**train_dl_kwargs)
    val_loader   = DataLoader(**val_dl_kwargs)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_classes=len(train_classes), target_size=target_size, logger=logger)
    model = model.to(device)

    # loss/optim/sched
    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)  # ✅ 一定要先建立，抓到正確 base_lrs
    
    # ✅ 讓 epoch1 用 base_lr / warmup_epochs
    warmup_epochs = int(cfg["train"]["warmup_epochs"])
    if warmup_epochs > 0:
        for pg, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            pg["lr"] = base_lr * (1.0 / warmup_epochs)

    # amp
    use_amp = bool(cfg["train"].get("amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # train loop
    epochs = int(cfg["train"]["epochs"])
    save_every = int(cfg["train"]["save_every"])
    f1_avg = cfg["eval"].get("f1_average", "macro")

    metrics = []
    logger.info("🟢 Start training: epochs=%d, amp=%s, device=%s", epochs, use_amp, device.type)

    t_begin = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, f1_avg=f1_avg)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.perf_counter() - t0

        logger.info(
            "Epoch %02d/%02d | TrainLoss %.4f | ValLoss %.4f | ValAcc %.2f%% | ValF1 %.4f | LR %.6f | Time %.2fs",
            epoch, epochs, train_loss, val_loss, val_acc, val_f1, current_lr, epoch_time
        )

        metrics.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "lr": float(current_lr),
            "epoch_time_sec": float(epoch_time),
        })

        # save_every <= 0 代表不存任何 checkpoint
        do_save = (save_every > 0) and ((epoch % save_every == 0) or (epoch == epochs))
        if do_save:
            save_checkpoint(out_dir, epoch, model, label_to_idx, logger)

        scheduler.step()

        # dump metrics every epoch (safe for long runs)
        metrics_path = out_dir / "metrics.jsonl"
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics[-1], ensure_ascii=False) + "\n")

    logger.info("✅ Done. Total time = %.2fs", time.perf_counter() - t_begin)
    logger.info("💾 end")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

