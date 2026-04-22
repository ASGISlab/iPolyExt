
from __future__ import annotations
import os, json, random, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import cv2, math
from PIL import Image, ImageEnhance
from tqdm import tqdm

try:
    import yaml
except ImportError as e:
    raise RuntimeError("請先安裝 pyyaml：pip install pyyaml") from e


# ----------------------------
# Config
# ----------------------------
@dataclass
class Cfg:
    seed: int
    big_root: Path
    small_root: Path
    out_root: Path

    opacity_weights: Dict[float, int]
    unit_per_weight: Optional[int]
    total_per_class: Optional[int]

    bg_weights: Dict[str, int]
    bg_code: Dict[str, str]

    max_retry_per_sample: int
    white_thr: int


def load_cfg(cfg_path: Path) -> Cfg:
    if isinstance(cfg_path, Path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
    else:
        d = cfg_path  # type: ignore

    def to_path(k: str) -> Path:
        return Path(d[k]).expanduser()

    # yaml key 是字串，轉 float
    opw = {float(k): int(v) for k, v in d["opacity_weights"].items()}

    unit = d.get("unit_per_weight", None)
    total = d.get("total_per_class", None)

    return Cfg(
        seed=int(d.get("seed", 0)),
        big_root=to_path("big_root"),
        small_root=to_path("small_root"),
        out_root=to_path("out_root"),
        opacity_weights=opw,
        unit_per_weight=None if unit in (None, "null") else int(unit),
        total_per_class=None if total in (None, "null") else int(total),
        bg_weights={str(k): int(v) for k, v in d["bg_weights"].items()},
        bg_code={str(k): str(v) for k, v in d["bg_code"].items()},
        max_retry_per_sample=int(d.get("max_retry_per_sample", 50)),
        white_thr=int(d.get("white_thr", 250)),
    )


# ----------------------------
# Utils
# ----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}


def _alloc_counts_from_weights(weights: Dict[Any, int], total: int) -> Dict[Any, int]:
    keys = list(weights.keys())
    w = np.array([weights[k] for k in keys], dtype=float)
    wsum = w.sum()
    raw = total * (w / wsum)

    base = np.floor(raw).astype(int)
    rem = total - int(base.sum())
    frac = raw - base

    order = np.argsort(-frac)
    for i in range(rem):
        base[order[i]] += 1

    return {k: int(base[idx]) for idx, k in enumerate(keys)}


def build_opacity_counts(op_weights: Dict[float, int],
                         unit_per_weight: Optional[int],
                         total_per_class: Optional[int]) -> Dict[float, int]:
    if unit_per_weight is not None:
        return {op: int(w * unit_per_weight) for op, w in op_weights.items()}
    if total_per_class is not None:
        return _alloc_counts_from_weights(op_weights, total_per_class)
    raise ValueError("unit_per_weight 與 total_per_class 必須至少指定一個")


def load_image_rgba(image_path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            times_w = math.ceil(98 / w)
            times_h = math.ceil(56 / h)
            if times_w>1 or times_h>1:
                new_width = w * times_w
                new_height = h * times_h
                
                # Create the new canvas
                tiled_img = Image.new("RGBA", (new_width, new_height))
                for i in range(times_w):
                    for j in range(times_h):
                        tiled_img.paste(img, (i * w, j * h))
                img = tiled_img
            return img.convert("RGBA")
    except Exception:
        return None


def is_image_white(image: Image.Image, thr: int = 250) -> bool:
    arr = np.asarray(image)
    return np.all(arr[:, :, :3] >= thr)


def reverse_add_reverse_bgra(img1_bgra: np.ndarray, img2_bgra: np.ndarray) -> np.ndarray:
    inv1 = cv2.bitwise_not(img1_bgra)
    inv2 = cv2.bitwise_not(img2_bgra)
    added = cv2.add(inv1, inv2)  # saturated add
    return cv2.bitwise_not(added)


def fade_to_white_bgra(img_bgra: np.ndarray, opacity: float) -> np.ndarray:
    bgr = img_bgra[..., :3]
    white = np.full_like(bgr, 255)
    faded_bgr = cv2.addWeighted(bgr, opacity, white, 1 - opacity, 0)
    out = img_bgra.copy()
    out[..., :3] = faded_bgr
    return out


def op_code(op: float) -> str:
    return f"{int(round(op * 100)):03d}"  # 1.0->100, 0.75->075...


def list_image_files(folder: Path):
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def quick_scan(root: Path):
    for p in sorted(root.iterdir()):
        if p.is_dir():
            cnt = sum(1 for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
            print(f"{p.name:<12} -> {cnt} images")

def has_images(folder: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in folder.iterdir())

def count_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)

# ----------------------------
# Core
# ----------------------------
class BGCache:
    def __init__(self, bg_paths: Dict[str, Path]):
        self.bg_paths = bg_paths
        self._cache: Dict[str, Image.Image] = {}

    def get(self, bg_name: str) -> Image.Image:
        if bg_name not in self._cache:
            with Image.open(self.bg_paths[bg_name]) as img:
                self._cache[bg_name] = img.convert("RGBA")
        return self._cache[bg_name]


def apply_translucent_overlay(
    *,
    cfg: Cfg,
    bg_cache: BGCache,
    bg_paths: Dict[str, Path],
    small_image: Image.Image,
    small_image_path: Path,
    save_dir: Path,
):
    # 1) opacity -> counts
    opacity_counts = build_opacity_counts(
        cfg.opacity_weights,
        unit_per_weight=cfg.unit_per_weight,
        total_per_class=cfg.total_per_class,
    )

    small_label = small_image_path.stem
    dst = save_dir / small_label
    dst.mkdir(parents=True, exist_ok=True)

    # 2) 逐 opacity 產出
    for op, n_gen in opacity_counts.items():
        if n_gen <= 0:
            continue

        # op=0: 不加背景
        if op == 0.0:
            for idx in range(n_gen):
                out_path = dst / f"{small_label}__bg-NONE__op-000__{idx}.png"
                small_image.save(out_path)
            continue

        # op!=0: BG 2:1:1 分配
        bg_counts = _alloc_counts_from_weights(cfg.bg_weights, n_gen)
        bg_schedule = []
        for bg_name, cnt in bg_counts.items():
            bg_schedule += [bg_name] * cnt
        random.shuffle(bg_schedule)

        # 逐張生成
        for idx in range(n_gen):
            bg_name = bg_schedule[idx]
            bg_code = cfg.bg_code.get(bg_name, "BG")
            large_image = bg_cache.get(bg_name)

            if large_image.width < small_image.width or large_image.height < small_image.height:
                raise RuntimeError(
                    f"大底圖尺寸不足：{bg_name} ({large_image.width}x{large_image.height}) "
                    f"< 小圖 ({small_image.width}x{small_image.height})"
                )

            forced_white_ok = False
            x = y = 0
            crop = None

            for attempt in range(cfg.max_retry_per_sample):
                x = random.randint(0, large_image.width - small_image.width)
                y = random.randint(0, large_image.height - small_image.height)
                crop = large_image.crop((x, y, x + small_image.width, y + small_image.height))
                crop = ImageEnhance.Color(crop).enhance(random.uniform(0.5, 1.5))

                if not is_image_white(crop, thr=cfg.white_thr):
                    break

                if attempt == cfg.max_retry_per_sample - 1:
                    forced_white_ok = True

            assert crop is not None

            # 合成
            fg_bgra = cv2.cvtColor(np.array(small_image), cv2.COLOR_RGBA2BGRA)
            bg_bgra = cv2.cvtColor(np.array(crop), cv2.COLOR_RGBA2BGRA)
            bg_bgra = fade_to_white_bgra(bg_bgra, op)
            combined_bgra = reverse_add_reverse_bgra(fg_bgra, bg_bgra)

            combined = Image.fromarray(cv2.cvtColor(combined_bgra, cv2.COLOR_BGRA2RGBA), "RGBA")

            # 你原本 ImageDataGenerator() 沒設參數，等同幾乎不變；這裡先保留「不改動」
            aug_img = combined

            fname = f"{small_label}__bg-{bg_code}__op-{op_code(op)}_{idx}.png"
            out_path = dst / fname
            aug_img.save(out_path)

            meta_path = dst / f"{Path(fname).stem}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "filename": out_path.name,
                        "label": small_label,
                        "opacity": op,
                        "bg_name": bg_name,
                        "bg_code": bg_code,
                        "crop_position": (x, y),
                        "forced_white_ok": forced_white_ok,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )


def process_subfolder(cfg: Cfg, bg_paths: Dict[str, Path], sub: Path):
    save_dir = cfg.out_root / sub.name
    save_dir.mkdir(parents=True, exist_ok=True)

    bg_cache = BGCache(bg_paths)

    imgs = list(list_image_files(sub))
    for small_path in imgs:
        small_img = load_image_rgba(small_path)
        if small_img is None:
            continue

        apply_translucent_overlay(
            cfg=cfg,
            bg_cache=bg_cache,
            bg_paths=bg_paths,
            small_image=small_img,
            small_image_path=small_path,
            save_dir=save_dir,
        )


def main(yaml_obj=None):
    if yaml_obj is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--cfg", type=str, required=True, help="path to yaml config")
        args = ap.parse_args()

        cfg = load_cfg(Path(args.cfg))
    else:
        cfg = load_cfg(yaml_obj)

    # seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # env (你原本 Colab 的 MKL 設定；這裡保留但不強迫)
    os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX")

    cfg.out_root.mkdir(parents=True, exist_ok=True)

    # 指定 3 張底圖（檢查存在）
    bg_paths: Dict[str, Path] = {}
    for fname in cfg.bg_weights.keys():
        p = cfg.big_root / fname
        if not p.exists():
            raise RuntimeError(f"指定的大底圖不存在：{p}")
        bg_paths[fname] = p

    print("✅ 指定大底圖 OK：")
    for k, v in bg_paths.items():
        print(" -", k, "->", v)

    print("\n小圖各資料夾張數：")
    if has_images(cfg.small_root):
        # small_root 本身就是一個類別資料夾（例如 .../XIV）
        print(f"{cfg.small_root.name:<12} -> {count_images(cfg.small_root)} images")
        targets = [cfg.small_root]
    else:
        # small_root 底下有多個類別資料夾（例如 II/IV/.../XIV）
        quick_scan(cfg.small_root)
        targets = [p for p in sorted(cfg.small_root.iterdir()) if p.is_dir()]

    for sub in tqdm(targets, desc="Folders"):
        process_subfolder(cfg, bg_paths, sub)


    print("\n✅ 全部處理完畢！輸出位於：", str(cfg.out_root))


if __name__ == "__main__":
    main()
