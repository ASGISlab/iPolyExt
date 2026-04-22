from pathlib import Path
import sys, subprocess, json, re, shutil
import yaml
from datetime import datetime
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
from src.augment_and_train_classifier.train_legend_classifier import main as train_legend_classifier_main
current_map = config.args.map
method = "GeoNet_Orig_augmentation"

# =========================
# 0) 你已有 current_map 就不用改；沒有就自己先設
# =========================
# current_map = "XIV"

# =========================
# 1) script / cfg
# =========================

cfg    = Path("configs/augment_and_train_classifier/train_legend_classifier.yaml")
assert cfg.is_file(), f"cfg not found: {cfg}"

# =========================
# 2) 找最新 fixed_train_data_YYYYMMDD_HHMMSS
# =========================
stage3_root = Path("data/train_data/stage_3")/method / str(current_map)
model_root  = Path("model")/method / str(current_map)

assert stage3_root.exists(), f"stage3_root not found: {stage3_root}"

pat = re.compile(r"^fixed_train_data_(\d{8})_(\d{6})$")
cands = [p for p in stage3_root.iterdir() if p.is_dir() and p.name.startswith("fixed_train_data_")]
if not cands:
    raise FileNotFoundError(f"找不到任何 fixed_train_data_* 在：{stage3_root}")

def parse_dt(p: Path):
    m = pat.match(p.name)
    return datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S") if m else None

def sort_key(p: Path):
    dt = parse_dt(p)
    return (1, dt.timestamp()) if dt else (0, p.stat().st_mtime)

latest_dir = max(cands, key=sort_key)
print("✅ latest fixed_train_data =", latest_dir)

# 你的資料夾結構：fixed_train_data_.../train  &  /val
train_dir = latest_dir / "train"
val_dir   = latest_dir / "val"
if not train_dir.exists(): raise FileNotFoundError(f"train_dir not found: {train_dir}")
if not val_dir.exists():   raise FileNotFoundError(f"val_dir not found: {val_dir}")

# output_dir 用同名固定資料包，方便追溯
output_dir = model_root / latest_dir.name
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 3) overrides（只覆寫 paths，其它照 yaml）
# =========================
overrides = {
    "paths": {
        "base_dir": str(latest_dir),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "output_dir": str(output_dir),
    }
}

def deep_update(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v

with open(cfg, "r", encoding="utf-8") as f:
    d = yaml.safe_load(f)

deep_update(d, overrides)

print(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))

# run_info
run_info = {
    "time": datetime.now().isoformat(),
    "current_map": str(current_map),
    "data_fixed_dir": str(latest_dir),
    "train_dir": str(train_dir),
    "val_dir": str(val_dir),
    "cfg_base": str(cfg),
}
with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
    json.dump(run_info, f, ensure_ascii=False, indent=2)
print("📝 wrote:", output_dir / "run_info.json")

# =========================
# 4) 執行訓練
# =========================
train_legend_classifier_main(d)


# ============================================================
# 5) 訓練完：自動選 best epoch（保證一定選得到）+ 只留 best_model
#    新版：用「val_loss 谷底帶」挑泛化 + 再套你的人眼規則 train_loss<=0.04
# ============================================================

EPOCH_LIMIT = 35

# 你對 F1 的偏好
TARGET_F1  = 0.995   # 目標（同時當上限附近的參考）
F1_CAP     = 0.995   # 絕對不要超過這個（避免後期虛高）
PREFER_F1  = 0.98
MIN_F1     = 0.97

# 穩定性（避免單點矇到）
STABLE_K = 3
F1_DROP_TOL = 0.01

# 你的人眼收斂條件
TRAIN_LOSS_MAX = 0.04  # 正常情況必須 <=0.04
TRAIN_LOSS_RELAX = 0.06  # 若找不到才放寬（保證一定選得到）

# val_loss 谷底帶（越小越嚴格）
VAL_LOSS_BAND = 0.10   # 10%：val_loss <= min_val_loss * (1+0.10)

DELETE_OTHERS = True

metrics_path = output_dir / "metrics.jsonl"
ckpt_dir = output_dir / "checkpoints"
if not metrics_path.is_file():
    raise FileNotFoundError(f"metrics.jsonl not found: {metrics_path}")
if not ckpt_dir.is_dir():
    raise FileNotFoundError(f"checkpoints dir not found: {ckpt_dir}")

# ---- 讀 metrics ----
rows = []
with open(metrics_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
rows = sorted(rows, key=lambda x: x["epoch"])
max_epoch = rows[-1]["epoch"]
by_epoch = {r["epoch"]: r for r in rows}
epoch_limit = min(EPOCH_LIMIT, max_epoch)
print(f"✅ loaded metrics: epochs=1..{max_epoch}")
print(f"🧯 epoch_limit = {epoch_limit} (EPOCH_LIMIT={EPOCH_LIMIT})")

# ---- helpers ----
def stable_after(e, limit):
    f0 = by_epoch[e]["val_f1"]
    e2 = min(limit, e + STABLE_K)
    future = [by_epoch[k]["val_f1"] for k in range(e, e2 + 1) if k in by_epoch]
    return (min(future) >= f0 - F1_DROP_TOL)

def eligible_base(r, require_stable, limit):
    # F1 上限先硬卡住（避免你說的 0.995+ 虛高）
    if r["val_f1"] > F1_CAP:
        return False
    if require_stable and (not stable_after(r["epoch"], limit)):
        return False
    return True

# ---- 1) 找 val_loss_min (在 1..epoch_limit 內) ----
val_losses = [by_epoch[e]["val_loss"] for e in range(1, epoch_limit + 1) if e in by_epoch]
val_loss_min = min(val_losses)
band_thr = val_loss_min * (1.0 + VAL_LOSS_BAND)
print(f"📉 val_loss_min={val_loss_min:.6f}, band_thr(+(%))={band_thr:.6f} (VAL_LOSS_BAND={VAL_LOSS_BAND})")

def in_val_band(r):
    return (r["val_loss"] <= band_thr + 1e-12)

# ---- 2) 建候選：先要求 val_loss 在谷底帶 + F1 不超 cap + 穩定性 ----
cands = []
for e in range(1, epoch_limit + 1):
    if e not in by_epoch:
        continue
    r = dict(by_epoch[e])
    r["epoch"] = e
    if not eligible_base(r, require_stable=True, limit=epoch_limit):
        continue
    if not in_val_band(r):
        continue
    cands.append(r)

print(f"✅ candidates in val-loss band (stable, f1<=cap): {len(cands)}")

# ---- 3) 先套 train_loss<=0.04；若完全沒有，放寬到 0.06；再沒有就不限制 ----
def filter_by_train_loss(cands, tl_max):
    return [r for r in cands if r["train_loss"] <= tl_max + 1e-12]

cands_tl = filter_by_train_loss(cands, TRAIN_LOSS_MAX)
used_tl = TRAIN_LOSS_MAX
if len(cands_tl) == 0:
    cands_tl = filter_by_train_loss(cands, TRAIN_LOSS_RELAX)
    used_tl = TRAIN_LOSS_RELAX
if len(cands_tl) == 0:
    cands_tl = cands[:]   # 最終保底：不限制 train_loss（但仍在 val_loss band 且穩定）
    used_tl = None

print(f"✅ candidates after train_loss filter: {len(cands_tl)} (train_loss_max_used={used_tl})")

# ---- 4) 選法 B：「達標就收」：在候選中找最早達到 0.98；不行就 0.97；再不行挑最接近 0.995(不超過) ----
def pick_earliest_f1_at_least(cands, thr):
    xs = [r for r in cands if r["val_f1"] >= thr]
    if not xs:
        return None
    return min(xs, key=lambda r: r["epoch"])

def pick_closest_to_target_under_cap(cands):
    # 優先 <=TARGET；同距離優先不超過；再來越早
    best = None
    best_key = None
    for r in cands:
        f1 = r["val_f1"]
        key = (abs(f1 - TARGET_F1), f1 > TARGET_F1, r["epoch"])
        if best_key is None or key < best_key:
            best_key = key
            best = r
    return best

best = pick_earliest_f1_at_least(cands_tl, PREFER_F1)
reason = f"earliest in val_loss band with stable f1>= {PREFER_F1}, f1<=cap, train_loss_max={used_tl}"
if best is None:
    best = pick_earliest_f1_at_least(cands_tl, MIN_F1)
    reason = f"earliest in val_loss band with stable f1>= {MIN_F1}, f1<=cap, train_loss_max={used_tl}"
if best is None:
    best = pick_closest_to_target_under_cap(cands_tl)
    reason = f"closest to {TARGET_F1} in val_loss band with stability (fallback), train_loss_max={used_tl}"

# ---- 最終保底：若 val_loss band 候選竟然為 0（極少見），就退回全區間（仍維持 f1<=cap + stability） ----
if best is None:
    all_cands = []
    for e in range(1, epoch_limit + 1):
        if e not in by_epoch:
            continue
        r = dict(by_epoch[e])
        r["epoch"] = e
        if not eligible_base(r, require_stable=True, limit=epoch_limit):
            continue
        all_cands.append(r)

    # 先套 train_loss<=0.04（不行再放寬）
    all_tl = filter_by_train_loss(all_cands, TRAIN_LOSS_MAX)
    used_tl2 = TRAIN_LOSS_MAX
    if len(all_tl) == 0:
        all_tl = filter_by_train_loss(all_cands, TRAIN_LOSS_RELAX)
        used_tl2 = TRAIN_LOSS_RELAX
    if len(all_tl) == 0:
        all_tl = all_cands[:]
        used_tl2 = None

    best = pick_earliest_f1_at_least(all_tl, PREFER_F1) or pick_earliest_f1_at_least(all_tl, MIN_F1) or pick_closest_to_target_under_cap(all_tl)
    reason = f"NO val_loss-band candidates -> fallback to all epochs<= {epoch_limit} (stable, f1<=cap), train_loss_max={used_tl2}"

if best is None:
    raise RuntimeError("Unexpected: cannot pick any epoch (metrics parsing broken?)")

best_epoch = int(best["epoch"])
r_best = by_epoch[best_epoch]

print("✅ Selected epoch:", best_epoch, "| reason:", reason)
print("   val_f1 =", r_best["val_f1"], "train_loss =", r_best["train_loss"], "val_loss =", r_best["val_loss"])



# ---- 複製成 best_model.pth（英文固定名） ----
src_ckpt = ckpt_dir / f"model_epoch_{best_epoch:02}.pth"
if not src_ckpt.is_file():
    raise FileNotFoundError(f"Selected checkpoint not found: {src_ckpt}")

best_path = output_dir / "best_model.pth"
shutil.copy2(src_ckpt, best_path)
print("💾 wrote:", best_path)


# ---- 記錄 selection 結果（新版：不再使用 OVERFIT_*）----
select_info = {
    "time": datetime.now().isoformat(),
    "selected_epoch": best_epoch,
    "epoch_limit": epoch_limit,

    "threshold_prefer": PREFER_F1,
    "threshold_min": MIN_F1,
    "target_f1": TARGET_F1,
    "f1_cap": F1_CAP,
    "reason": reason,

    "stable_k": STABLE_K,
    "f1_drop_tol": F1_DROP_TOL,

    "train_loss_max": TRAIN_LOSS_MAX,
    "train_loss_relax": TRAIN_LOSS_RELAX,
    "train_loss_max_used": used_tl,          # 可能是 0.04 / 0.06 / None
    "val_loss_band": VAL_LOSS_BAND,
    "val_loss_min": float(val_loss_min),
    "val_loss_band_thr": float(band_thr),
    "num_candidates_in_band": int(len(cands)),
    "num_candidates_after_train_loss": int(len(cands_tl)),

    "metrics_at_selected": r_best,
    "source_checkpoint": str(src_ckpt),
    "best_model_path": str(best_path),
}

with open(output_dir / "best_model_info.json", "w", encoding="utf-8") as f:
    json.dump(select_info, f, ensure_ascii=False, indent=2)
print("📝 wrote:", output_dir / "best_model_info.json")


# ---- 刪除 checkpoints（含資料夾本體）----
ckpt_dir = output_dir / "checkpoints"
best_path = output_dir / "best_model.pth"

# 1) 保險：best_model.pth 一定要存在且非空
if (not best_path.is_file()) or (best_path.stat().st_size == 0):
    raise RuntimeError(f"best_model.pth 不存在或是空檔，取消刪除 checkpoints：{best_path}")

# 2) 刪除 checkpoints 內所有檔案
if ckpt_dir.is_dir():
    rm = 0
    for p in ckpt_dir.glob("*"):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                rm += 1
            elif p.is_dir():
                shutil.rmtree(p)
                rm += 1
        except Exception as e:
            print("⚠️ failed to delete:", p, "err=", e)

    # 3) 刪除 checkpoints 資料夾本體（若已空）
    try:
        ckpt_dir.rmdir()  # 只有空資料夾才能 rmdir
        print(f"🧹 deleted checkpoints folder: {ckpt_dir} (removed items={rm})")
    except OSError:
        # 代表資料夾非空或其他問題，就用 rmtree 強刪
        shutil.rmtree(ckpt_dir)
        print(f"🧹 force deleted checkpoints folder: {ckpt_dir} (removed items={rm})")
else:
    print("ℹ️ checkpoints dir not found, skip:", ckpt_dir)
