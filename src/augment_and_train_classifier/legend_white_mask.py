
"""
White occlusion augmentation (single target distribution).
- Traverse: src_root / {sheet} / {class} / *.png
- Per-class exact allocation of noise modes (line/rect/none) based on weights
- Single target area distribution: target/min/max
- Write tag into filename: __nz-{tag} inserted before the last index if exists
"""

import os
import re
import json, math
import shutil
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ============================================================
# Config
# ============================================================
@dataclass
class AugConfig:
    src_root: Path
    out_root: Path
    overwrite_out_root: bool = True
    copy_json: bool = False
    img_exts: Tuple[str, ...] = (".png",)

    seed: int = 12345

    noise_mode_weights: Dict[str, int] = None
    noise_area_target_single: Dict[str, float] = None
    cand_tries_num: int = 6

    noise_tag: Dict[str, str] = None

    # line mask
    line_mask: Dict = None
    # rect/blob mask
    rect_mask: Dict = None

    # union/erode settings
    union_until: Dict = None
    erode_until: Dict = None


def load_config(path: Path) -> AugConfig:
    if isinstance(path, Path):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = path
    cfg = AugConfig(
        src_root=Path(data["src_root"]),
        out_root=Path(data["out_root"]),
        overwrite_out_root=bool(data.get("overwrite_out_root", True)),
        copy_json=bool(data.get("copy_json", False)),
        img_exts=tuple([s.lower() for s in data.get("img_exts", [".png"])]),
        seed=int(data.get("seed", 12345)),
        noise_mode_weights=data.get("noise_mode_weights", {"line": 1, "rect": 1, "none": 1}),
        noise_area_target_single=data.get("noise_area_target_single", {"target": 0.21, "min": 0.07, "max": 0.36}),
        cand_tries_num=int(data.get("cand_tries_num", 6)),
        noise_tag=data.get("noise_tag", {"none": "N", "line": "L", "rect": "R"}),
        line_mask=data.get("line_mask", {}),
        rect_mask=data.get("rect_mask", {}),
        union_until=data.get("union_until", {}),
        erode_until=data.get("erode_until", {}),
    )
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Utilities
# ============================================================
def alloc_counts_from_weights(weights: dict, total: int) -> dict:
    """Allocate integer counts that sum to total, proportionally to weights."""
    keys = list(weights.keys())
    w = np.array([weights[k] for k in keys], dtype=float)
    raw = total * (w / w.sum())
    base = np.floor(raw).astype(int)
    rem = total - int(base.sum())
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(rem):
        base[order[i]] += 1
    return {k: int(base[i]) for i, k in enumerate(keys)}


def insert_tag_before_last_index(stem: str, tag: str) -> str:
    """
    Insert __nz-xxx before last index if the stem ends with __<digits> or _<digits>.
    1) op-000__5 -> op-000__nz-tag__5
    2) op-075_5  -> op-075__nz-tag_5
    3) else      -> stem__nz-tag
    """
    m = re.match(r"^(.*)__(\d+)$", stem)
    if m:
        base, idx = m.group(1), m.group(2)
        return f"{base}__nz-{tag}__{idx}"

    m = re.match(r"^(.*)_(\d+)$", stem)
    if m:
        base, idx = m.group(1), m.group(2)
        return f"{base}__nz-{tag}_{idx}"

    return f"{stem}__nz-{tag}"


# ============================================================
# Mask generators (same behavior as your notebook)
# ============================================================
def random_brush_mask(
    h, w,
    min_strokes=1, max_strokes=2,
    max_vertices=5, max_len=256,
    max_brush_width=10, max_angles=360,
    area_limit=0.6,
    max_tries=10
):
    limit_pixels = int(area_limit * h * w)

    for _ in range(max_tries):
        mask = np.zeros((h, w), np.uint8)
        num_strokes = random.randint(min_strokes, max_strokes)
        for _ in range(num_strokes):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            angle = random.uniform(0, 2 * np.pi)
            num_vertices = random.randint(1, max_vertices)
            for _ in range(num_vertices):
                angle += random.uniform(-max_angles, max_angles) * np.pi / 180
                length = random.randint(80, max_len)
                brush_w = random.randint(6, max_brush_width)
                x_n = np.clip(x + length * np.cos(angle), 0, w - 1)
                y_n = np.clip(y + length * np.sin(angle), 0, h - 1)
                cv2.line(mask, (int(x), int(y)), (int(x_n), int(y_n)), 255, brush_w)
                x, y = x_n, y_n

        k = random.choice([3, 5, 7])
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = (mask > 127).astype(np.uint8)

        if mask.sum() <= limit_pixels:
            return mask

    # if exceeded: erode
    excess = mask.sum() - limit_pixels
    if excess > 0:
        erode_iter = max(1, int(excess / (0.05 * h * w)))
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=erode_iter)
    return (mask > 0).astype(np.uint8)


def random_blob_mask(
    h, w,
    min_frac=0.15, max_frac=0.35,
    max_polys=5, min_v=6, max_v=12,
    area_limit=0.6, max_tries=10
):
    limit_pixels = int(area_limit * h * w)

    for _ in range(max_tries):
        mask = np.zeros((h, w), np.uint8)
        num_polys = random.randint(1, max_polys)
        target_a = random.uniform(min_frac, max_frac) * h * w

        for _ in range(num_polys):
            cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
            radius = int(np.sqrt(target_a / (np.pi * num_polys)))
            n_vert = random.randint(min_v, max_v)
            angles = np.sort(np.random.rand(n_vert) * 2 * np.pi)
            pts = []
            for ang in angles:
                r = radius * random.uniform(0.8, 1.3)
                x = np.clip(cx + r * np.cos(ang), 0, w - 1)
                y = np.clip(cy + r * np.sin(ang), 0, h - 1)
                pts.append([int(x), int(y)])
            cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = (mask > 127).astype(np.uint8)

        if mask.sum() <= limit_pixels:
            return mask

    # if exceeded: erode
    excess = mask.sum() - limit_pixels
    if excess > 0:
        erode_iter = max(1, int(excess / (0.05 * h * w)))
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=erode_iter)
    return (mask > 0).astype(np.uint8)


def erode_until(mask01: np.ndarray, target_max_frac: float, max_iter: int = 30) -> np.ndarray:
    h, w = mask01.shape
    max_pixels = int(target_max_frac * h * w)
    out = mask01.copy()
    it = 0
    while out.sum() > max_pixels and it < max_iter:
        out = cv2.erode(out, np.ones((3, 3), np.uint8), iterations=1)
        it += 1
        if out.sum() == 0:
            break
    return (out > 0).astype(np.uint8)


def union_until(
    mask01: np.ndarray,
    gen_fn,
    target_min_frac: float,
    target_max_frac: float,
    max_add: int = 5,
    erode_max_iter: int = 30
) -> np.ndarray:
    """If too small, union more; if too large, erode back."""
    h, w = mask01.shape
    min_pixels = int(target_min_frac * h * w)
    out = mask01.copy()
    add = 0
    while out.sum() < min_pixels and add < max_add:
        extra = gen_fn()
        out = np.maximum(out, extra)
        add += 1
        if out.sum() == 0:
            break

    if out.sum() > int(target_max_frac * h * w):
        out = erode_until(out, target_max_frac, max_iter=erode_max_iter)

    return (out > 0).astype(np.uint8)


def make_noise_mask(mode: str, h: int, w: int, cfg: AugConfig) -> np.ndarray:
    """Single target distribution; pick best among candidates."""
    area_cfg = cfg.noise_area_target_single
    tmin, tmax, tgt = area_cfg["min"], area_cfg["max"], area_cfg["target"]
    CAND_TRIES = cfg.cand_tries_num

    erode_max_iter = int(cfg.erode_until.get("max_iter", 30))

    if mode == "line":
        lm = cfg.line_mask
        max_len = max(h, w) if lm.get("max_len_mode", "max_hw") == "max_hw" else int(lm.get("max_len", max(h, w)))

        params = dict(
            min_strokes=int(lm.get("min_strokes", 1)),
            max_strokes=int(lm.get("max_strokes", 3)),
            max_brush_width=int(lm.get("max_brush_width", 14)),
            max_len=int(max_len),
            area_limit=float(tmax),
            max_tries=int(lm.get("max_tries", 10))
        )

        max_add = int(cfg.union_until.get("max_add_line", 6))

        def gen_once():
            base = random_brush_mask(h, w, **params)
            return union_until(
                base,
                lambda: random_brush_mask(h, w, **params),
                tmin, tmax,
                max_add=max_add,
                erode_max_iter=erode_max_iter
            )

    elif mode == "rect":
        rm = cfg.rect_mask
        min_frac = max(0.02, tgt * 0.8)
        max_frac = min(0.60, tgt * 1.6)

        params = dict(
            min_frac=float(min_frac),
            max_frac=float(max_frac),
            max_polys=int(rm.get("max_polys", 4)),
            area_limit=float(tmax),
            max_tries=int(rm.get("max_tries", 10))
        )

        max_add = int(cfg.union_until.get("max_add_rect", 4))

        def gen_once():
            base = random_blob_mask(h, w, **params)
            return union_until(
                base,
                lambda: random_blob_mask(h, w, **params),
                tmin, tmax,
                max_add=max_add,
                erode_max_iter=erode_max_iter
            )

    else:
        raise ValueError(f"Unknown mode for mask: {mode}")

    best = None
    best_score = 1e9
    for _ in range(CAND_TRIES):
        m = gen_once()
        frac = float(m.mean())
        score = abs(frac - tgt)
        if score < best_score:
            best_score = score
            best = m
            if best_score <= 0.01:
                break
    return best


def apply_mask_rgb(img_rgb: np.ndarray, mask01: np.ndarray, fill_value: int = 255) -> np.ndarray:
    out = img_rgb.copy()
    out[mask01 == 1] = [fill_value, fill_value, fill_value]
    return out


# ============================================================
# Main
# ============================================================

def run(cfg: AugConfig):
    set_seed(cfg.seed)

    # 永遠不要刪整個 out_root
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    stat_mode = {"none": 0, "line": 0, "rect": 0}

    # ✅ 你要保護的類別：遇到就「不刪、不改、不覆蓋」
    PRESERVE = {"blank", "bgc"}  # 用小寫做比對（case-insensitive）

    def _has_images_direct(d: Path, exts: Tuple[str, ...]) -> bool:
        return any(p.is_file() and p.suffix.lower() in exts for p in d.iterdir())

    # 判斷 src_root 是不是「單一 sheet」：src_root/{class}/*.png
    sheet_dirs_guess = sorted([p for p in cfg.src_root.iterdir() if p.is_dir()])
    if sheet_dirs_guess and any(_has_images_direct(sd, cfg.img_exts) for sd in sheet_dirs_guess):
        sheet_dirs = [cfg.src_root]        # 單一 sheet
    else:
        sheet_dirs = sheet_dirs_guess      # 多 sheet

    for sheet_dir in sheet_dirs:
        # ✅ 輸出永遠放在 out_root/sheet_name/
        out_sheet_dir = cfg.out_root / sheet_dir.name
        out_sheet_dir.mkdir(parents=True, exist_ok=True)

        # ✅ overwrite_out_root=True 時：只清掉「非 blank/BGC」的資料夾
        if cfg.overwrite_out_root:
            for sub in out_sheet_dir.iterdir():
                if not sub.is_dir():
                    continue
                if sub.name.lower() in PRESERVE:
                    continue  # 保留 blank/BGC
                shutil.rmtree(sub)

        class_dirs = sorted([p for p in sheet_dir.iterdir() if p.is_dir()])
        for cls_dir in class_dirs:
            cls_name = cls_dir.name

            # ✅ 直接跳過 blank/BGC：不處理、不覆蓋，完全不碰
            if cls_name.lower() in PRESERVE:
                continue

            pngs = sorted([p for p in cls_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in cfg.img_exts])
            if not pngs:
                continue

            total_in += len(pngs)

            mode_counts = alloc_counts_from_weights(cfg.noise_mode_weights, len(pngs))

            schedule: List[str] = []
            schedule += ["none"] * mode_counts.get("none", 0)
            schedule += ["line"] * mode_counts.get("line", 0)
            schedule += ["rect"] * mode_counts.get("rect", 0)
            assert len(schedule) == len(pngs), "schedule length != number of images (should not happen)"

            random.shuffle(pngs)
            random.shuffle(schedule)

            out_cls_dir = out_sheet_dir / cls_name
            out_cls_dir.mkdir(parents=True, exist_ok=True)

            for p, mode in tqdm(list(zip(pngs, schedule)),
                                desc=f"{sheet_dir.name}/{cls_name}", leave=False):
                img0 = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img0 is None:
                    print(f"[Warn] Failed to read: {p}")
                    continue
                h, w = img0.shape[:2]

                
                times_w = math.ceil(98 / w)
                times_h = math.ceil(56 / h)
                if times_w > 1 or times_h > 1:
                    # np.tile expects (height_reps, width_reps, channel_reps)
                    # If it's a color image, shape is 3. If grayscale, shape is 2.
                    reps = (times_h, times_w, 1) if len(img0.shape) == 3 else (times_h, times_w)
                    
                    img0 = np.tile(img0, reps)

                if img0.ndim == 2:
                    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)

                if img0.shape[2] == 4:
                    bgr = img0[:, :, :3]
                    a = img0[:, :, 3]
                else:
                    bgr = img0
                    a = None

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]

                if mode == "none":
                    white_pct = 0
                    tag = f"{cfg.noise_tag[mode]}{white_pct:02d}"
                    rgb2 = rgb
                    stat_mode["none"] += 1
                else:
                    mask = make_noise_mask(mode, h, w, cfg)
                    white_frac = float(mask.mean())
                    white_pct = int(round(white_frac * 100))
                    tag = f"{cfg.noise_tag[mode]}{white_pct:02d}"
                    rgb2 = apply_mask_rgb(rgb, mask, fill_value=255)
                    stat_mode[mode] += 1

                new_stem = insert_tag_before_last_index(p.stem, tag)
                out_path = out_cls_dir / f"{new_stem}{p.suffix}"

                bgr2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR)
                out_img = np.dstack([bgr2, a]) if a is not None else bgr2
                ok = cv2.imwrite(str(out_path), out_img)
                if not ok:
                    print(f"[Warn] Failed to write: {out_path}")
                    continue

                total_out += 1

                if cfg.copy_json:
                    src_json = p.with_suffix(".json")
                    if src_json.exists():
                        try:
                            data = json.loads(src_json.read_text(encoding="utf-8"))
                        except Exception:
                            data = {}
                    else:
                        data = {}

                    data.update({
                        "noise_mode": mode,
                        "noise_white_pct": white_pct,
                        "noise_tag": tag,
                        "filename": out_path.name,
                    })
                    out_json = out_path.with_suffix(".json")
                    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"[{sheet_dir.name}/{cls_name}] in={len(pngs)} -> out={len(pngs)} | mode_counts={mode_counts}")

    print("\n✅ Done")
    print(f"Total input images : {total_in}")
    print(f"Total output images: {total_out}")
    print("Mode stats:", stat_mode)
    print("Output root:", cfg.out_root)



def main(yaml_obj=None):
    import argparse
    if yaml_obj is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfg", type=str, required=True, help="path to yaml config")
        args = ap.parse_args()

        cfg = load_config(Path(args.cfg))
    else:
        cfg = load_config(yaml_obj)
    run(cfg)


if __name__ == "__main__":
    main()
