#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢驗單輪分割 + 分類結果是否有像素遺失。

整合三個原始 cell：
1) 錯誤整合圖：把 SAM分割錯誤 下的 tile / mid 貼回完整圖
2) 正確整合圖：把 SAM分割正確/merged 轉成 correct_only_full
3) 錯誤 + 正確整合圖：做 union，並檢查是否仍有缺失像素

使用方式：
    python check_round_integrity.py --config /path/to/check_round_integrity.yaml

設計原則：
- 盡量保留你目前 Jupyter cell 的邏輯與輸出命名
- 所有硬編碼路徑抽到 yaml
- 最後缺失檢查支援不同 target mask 模式，預設改為 original_image_nonwhite
  以符合「檢查整輪有沒有丟像素」的需求
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# =========================
# 基本工具
# =========================

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_obj(obj: Any, ctx: Dict[str, Any]) -> Any:
    """遞迴把 yaml 裡的字串做 .format(**ctx) 展開。"""
    if isinstance(obj, dict):
        return {k: format_obj(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [format_obj(v, ctx) for v in obj]
    if isinstance(obj, str):
        return obj.format(**ctx)
    return obj


def as_path(x: str | Path) -> Path:
    return x if isinstance(x, Path) else Path(x)


# =========================
# 影像工具
# =========================

def is_white(rgb: np.ndarray, white_value: int = 255) -> np.ndarray:
    return (
        (rgb[..., 0] == white_value)
        & (rgb[..., 1] == white_value)
        & (rgb[..., 2] == white_value)
    )


def to_rgb_u8(img: np.ndarray | None) -> np.ndarray:
    """
    支援灰階 / BGR / BGRA / 16-bit tif，回傳 RGB uint8。
    BGRA 會先合到白底。
    """
    if img is None:
        raise RuntimeError("讀圖失敗")

    if img.dtype != np.uint8:
        img_f = img.astype(np.float32)
        mx = float(img_f.max()) if img_f.size else 255.0
        mx = mx if mx > 0 else 255.0
        img = np.clip(img_f / mx * 255.0, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)

    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        a = (a.astype(np.float32) / 255.0)[..., None]
        rgb = cv2.merge([r, g, b]).astype(np.float32)
        out = np.clip(rgb * a + 255.0 * (1 - a), 0, 255).astype(np.uint8)
        return out

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_rgb(path: Path) -> np.ndarray:
    return to_rgb_u8(cv2.imread(str(path), cv2.IMREAD_UNCHANGED))


def save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def save_mask(path: Path, mask_bool: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), (mask_bool.astype(np.uint8) * 255))


def pad_white(img: np.ndarray, H: int, W: int) -> np.ndarray:
    out = np.full((H, W, 3), 255, np.uint8)
    h, w = img.shape[:2]
    out[:h, :w] = img
    return out


def resolve_image_path(p: Path, exts: Iterable[str]) -> Path:
    """
    p 可以是：
    - 完整檔名
    - 不含副檔名的 stem
    - 資料夾（取第一張符合副檔名的圖）
    """
    if p.exists():
        if p.is_file():
            return p
        if p.is_dir():
            hits: List[Path] = []
            for e in exts:
                hits += sorted(p.glob(f"*{e}"))
                hits += sorted(p.glob(f"*{e.upper()}"))
            if hits:
                return hits[0]
            raise FileNotFoundError(f"資料夾內找不到底圖：{p} (exts={tuple(exts)})")

    parent = p.parent
    stem = p.stem if p.suffix else p.name
    base = p.with_suffix("")

    for e in exts:
        for cand in (Path(str(base) + e), Path(str(base) + e.upper())):
            if cand.exists():
                return cand

    if parent.exists():
        for e in exts:
            for cand in (parent / f"{stem}{e}", parent / f"{stem}{e.upper()}"):
                if cand.exists():
                    return cand

    raise FileNotFoundError(f"找不到底圖：{p} (嘗試 exts={tuple(exts)})")


# =========================
# 錯誤整合圖
# =========================

def reconstruct_error_canvas(cfg: Dict[str, Any]) -> Dict[str, Any]:
    src_dir = as_path(cfg["error_src_dir"])
    out_dir = ensure_dir(as_path(cfg["error_out_dir"]))
    output_name = cfg.get("error_output_name", f"{src_dir.name}_reconstructed_full.png")
    recursive = bool(cfg.get("error_scan_recursive", False))
    white_value = int(cfg.get("white_value", 255))

    pattern = re.compile(
        r"^(?P<prefix>.+?)_ty(?P<ty>-?\d+)_tx(?P<tx>-?\d+)_(?P<tag>.+?)\.png$",
        re.IGNORECASE,
    )

    groups: Dict[str, Dict[Tuple[int, int], List[Path]]] = defaultdict(lambda: defaultdict(list))
    iterator = src_dir.rglob("*.png") if recursive else src_dir.iterdir()

    for p in iterator:
        if p.suffix.lower() != ".png":
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        prefix = m.group("prefix")
        ty = int(m.group("ty"))
        tx = int(m.group("tx"))
        groups[prefix][(ty, tx)].append(p)

    if not groups:
        raise RuntimeError(f"❌ 找不到符合命名規則的 png：{src_dir}")

    # 保留你原本 cell 的行為：把所有 prefix 合成同一張輸出圖
    merged: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    for tile_dict in groups.values():
        for k, vs in tile_dict.items():
            merged[k].extend(vs)

    tile_dict = dict(merged)

    h_max = 0
    w_max = 0
    max_hw_by_tile: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for (ty, tx), paths in tile_dict.items():
        mh = 0
        mw = 0
        for p in paths:
            img = read_rgb(p)
            h, w = img.shape[:2]
            mh = max(mh, h)
            mw = max(mw, w)
        max_hw_by_tile[(ty, tx)] = (mh, mw)
        h_max = max(h_max, ty + mh)
        w_max = max(w_max, tx + mw)

    canvas = np.full((h_max, w_max, 3), white_value, np.uint8)

    for (ty, tx), paths in tqdm(tile_dict.items(), desc="拼接錯誤圖"):
        mh, mw = max_hw_by_tile[(ty, tx)]
        tile = np.full((mh, mw, 3), white_value, np.uint8)

        paths_sorted = sorted(
            paths,
            key=lambda p: (0 if "composite_small" in p.name else 1, p.name),
        )

        for p in paths_sorted:
            img = read_rgb(p)
            h, w = img.shape[:2]
            patch = tile[0:h, 0:w]
            mask = ~is_white(img, white_value=white_value)
            patch[mask] = img[mask]

        yc = slice(ty, ty + mh)
        xc = slice(tx, tx + mw)
        targ = canvas[yc, xc]
        mask_tile = ~is_white(tile, white_value=white_value)
        targ[mask_tile] = tile[mask_tile]

    out_path = out_dir / output_name
    save_rgb(out_path, canvas)

    return {
        "error_reconstructed_path": str(out_path),
        "error_canvas_shape": [int(canvas.shape[0]), int(canvas.shape[1]), int(canvas.shape[2])],
        "error_foreground_pixels": int((~is_white(canvas, white_value=white_value)).sum()),
    }


# =========================
# 正確整合圖
# =========================

def mask_from_image(img: np.ndarray, white_th_pixel: int) -> np.ndarray:
    if img.ndim == 2:
        return img < white_th_pixel
    if img.shape[2] == 4:
        return img[:, :, 3] > 0
    return np.any(img[:, :, :3] < white_th_pixel, axis=2)


def reconstruct_correct_canvas(cfg: Dict[str, Any]) -> Dict[str, Any]:
    masks_dir = as_path(cfg["correct_masks_dir"])
    alt_image = as_path(cfg["alt_image"])
    out_root = ensure_dir(as_path(cfg["correct_out_dir"]))

    valid_exts = tuple(cfg.get("valid_exts", [".png", ".jpg", ".jpeg", ".tif", ".tiff"]))
    image_resolve_exts = tuple(cfg.get("image_resolve_exts", [".png", ".tif", ".tiff"]))
    white_th_pixel = int(cfg.get("white_th_pixel", 255))
    alpha = float(cfg.get("alpha", 0.5))
    random_seed = int(cfg.get("random_seed", 123))
    np.random.seed(random_seed)

    group = cfg.get("correct_group_name", masks_dir.name)
    pair_dir = ensure_dir(out_root / f"{group}_pairs_small_to_large")
    mid_dir = ensure_dir(out_root / f"{group}_mid_small_to_large")
    crop_dir = ensure_dir(out_root / f"{group}_crops_small_to_large")

    mask_files = [p for p in sorted(masks_dir.iterdir()) if p.suffix.lower() in valid_exts]
    if not mask_files:
        raise FileNotFoundError(f"在 {masks_dir} 找不到任何影像：{valid_exts}")

    sample = cv2.imread(str(mask_files[0]), cv2.IMREAD_UNCHANGED)
    if sample is None:
        raise RuntimeError(f"讀不到影像：{mask_files[0]}")
    H, W = sample.shape[:2]

    alt_path = resolve_image_path(alt_image, exts=image_resolve_exts)
    alt_raw = cv2.imread(str(alt_path), cv2.IMREAD_UNCHANGED)
    if alt_raw is None:
        raise RuntimeError(f"讀不到底圖：{alt_path}")
    alt_img = to_rgb_u8(alt_raw)

    if alt_img.shape[:2] != (H, W):
        raise ValueError(
            f"底圖尺寸不一致：alt_image={alt_img.shape[:2]}，"
            f"mask sample={(H, W)}，path={alt_path}"
        )
    base_image = alt_img.copy()

    final_masks: List[Dict[str, Any]] = []
    for p in tqdm(mask_files, desc=f"讀取/轉換 {group} 的 masks"):
        m_img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m_img is None:
            continue
        if m_img.shape[:2] != (H, W):
            raise ValueError(
                f"mask 尺寸不一致：got={m_img.shape[:2]}，expected={(H, W)}，path={p}"
            )
        m_bool = mask_from_image(m_img, white_th_pixel=white_th_pixel)
        area = int(m_bool.sum())
        if area == 0:
            continue
        final_masks.append({"mask": m_bool, "area": area, "src": p})

    final_masks.sort(key=lambda x: x["area"])
    if not final_masks:
        raise RuntimeError("所有 mask 都是空白，無可視化輸出。")

    recon = np.full_like(base_image, 255)
    summary_csv = out_root / f"{group}_summary.csv"

    with open(summary_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["idx", "area", "y0", "y1", "x0", "x1", "src_mask", "pair_png", "mid_png", "crop_png"])

        for idx, item in enumerate(tqdm(final_masks, desc="輸出正確圖視覺化"), start=1):
            mask_bool = item["mask"]
            ys, xs = np.where(mask_bool)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())

            color = np.random.randint(0, 255, 3, dtype=np.uint8)
            overlay = base_image.copy()
            ov = overlay[mask_bool].astype(np.float32)
            overlay[mask_bool] = (alpha * color + (1 - alpha) * ov).astype(np.uint8)

            mid = np.full_like(base_image, 255)
            mid[mask_bool] = base_image[mask_bool]

            recon[mask_bool] = base_image[mask_bool]
            recon_vis = recon.copy()

            combined = np.concatenate((overlay, mid, recon_vis), axis=1)

            pair_path = pair_dir / f"{group}_pair_{idx:03d}.png"
            mid_path = mid_dir / f"{group}_mid_{idx:03d}.png"
            crop_path = crop_dir / f"{group}_crop_{idx:03d}.png"

            save_rgb(pair_path, combined)
            save_rgb(mid_path, mid)

            crop = base_image[y0:y1 + 1, x0:x1 + 1].copy()
            crop_mask = mask_bool[y0:y1 + 1, x0:x1 + 1]
            crop[~crop_mask] = 255
            save_rgb(crop_path, crop)

            writer.writerow([
                idx,
                item["area"],
                y0,
                y1,
                x0,
                x1,
                str(item["src"]),
                str(pair_path),
                str(mid_path),
                str(crop_path),
            ])

    correct_only_path = out_root / cfg.get("correct_only_name", f"{group}_correct_only_full.png")
    correct_mask_path = out_root / cfg.get("correct_mask_name", f"{group}_correct_mask.png")

    save_rgb(correct_only_path, recon)
    correct_mask = ~np.all(recon == 255, axis=-1)
    save_mask(correct_mask_path, correct_mask)

    missing_vs_base = int(np.sum(np.any(recon != base_image, axis=-1)))

    return {
        "correct_only_path": str(correct_only_path),
        "correct_mask_path": str(correct_mask_path),
        "correct_summary_csv": str(summary_csv),
        "correct_group": group,
        "correct_mask_count": len(final_masks),
        "correct_foreground_pixels": int(correct_mask.sum()),
        "correct_missing_vs_base_pixels": missing_vs_base,
        "alt_image_resolved_path": str(alt_path),
    }


# =========================
# 錯誤 + 正確整合圖
# =========================

def build_target_mask(
    mode: str,
    img_correct: np.ndarray,
    img_error: np.ndarray,
    union_mask: np.ndarray,
    target_img: np.ndarray | None,
    white_value: int,
) -> np.ndarray:
    mode = mode.lower()

    if mode == "correct_image_nonwhite":
        return ~is_white(img_correct, white_value=white_value)

    if mode == "error_image_nonwhite":
        return ~is_white(img_error, white_value=white_value)

    if mode == "union_nonwhite":
        return union_mask.copy()

    if mode == "original_image_nonwhite":
        if target_img is None:
            raise ValueError("target_mask_mode=original_image_nonwhite 但未提供 target_img")
        return ~is_white(target_img, white_value=white_value)

    raise ValueError(
        "target_mask_mode 只支援："
        "correct_image_nonwhite / error_image_nonwhite / union_nonwhite / original_image_nonwhite"
    )

def union_correct_and_error(cfg: Dict[str, Any], prev: Dict[str, Any]) -> Dict[str, Any]:
    white_value = int(cfg.get("white_value", 255))

    correct_path = as_path(cfg.get("union_correct_path", prev["correct_only_path"]))
    error_path = as_path(cfg.get("union_error_path", prev["error_reconstructed_path"]))
    out_dir = ensure_dir(as_path(cfg["union_out_dir"]))
    alt_image_path = as_path(prev["alt_image_resolved_path"]) if prev.get("alt_image_resolved_path") else None
    
    out_union = out_dir / cfg.get("union_output_name", "union_correct_plus_error.png")
    out_missing_mask = out_dir / cfg.get("union_missing_mask_name", "union_missing_mask.png")
    out_missing_overlay = out_dir / cfg.get("union_missing_overlay_name", "union_with_missing_overlay.png")
    out_stats_json = out_dir / cfg.get("union_stats_json_name", "union_stats.json")
    
    img_c = read_rgb(correct_path)
    img_e = read_rgb(error_path)
    
    target_mode = cfg.get("target_mask_mode", "original_image_nonwhite")
    target_img = None
    
    # 先把 target 讀進來，不要等到 build_target_mask 裡才讀
    if target_mode == "original_image_nonwhite":
        if alt_image_path is None:
            raise ValueError("target_mask_mode=original_image_nonwhite 但未提供 alt_image_path")
        target_img = read_rgb(alt_image_path)
    
    # 三張一起決定最大畫布
    H = max(
        img_c.shape[0],
        img_e.shape[0],
        target_img.shape[0] if target_img is not None else 0,
    )
    W = max(
        img_c.shape[1],
        img_e.shape[1],
        target_img.shape[1] if target_img is not None else 0,
    )
    
    # 全部只做 pad_white，不做 resize
    if img_c.shape[:2] != (H, W):
        img_c = pad_white(img_c, H, W)
    if img_e.shape[:2] != (H, W):
        img_e = pad_white(img_e, H, W)
    if target_img is not None and target_img.shape[:2] != (H, W):
        target_img = pad_white(target_img, H, W)
    
    mask_c = ~is_white(img_c, white_value=white_value)
    mask_e = ~is_white(img_e, white_value=white_value)
    
    union_mask = mask_c | mask_e
    union_img = np.full((H, W, 3), 255, np.uint8)
    union_img[mask_c] = img_c[mask_c]
    fill_from_err = union_mask & (~mask_c)
    union_img[fill_from_err] = img_e[fill_from_err]
    
    target_mask = build_target_mask(
        mode=target_mode,
        img_correct=img_c,
        img_error=img_e,
        union_mask=union_mask,
        target_img=target_img,
        white_value=white_value,
    )

    missing_mask = target_mask & (~union_mask)
    missing_px = int(missing_mask.sum())
    total_target = int(target_mask.sum())
    missing_ratio = (missing_px / total_target * 100.0) if total_target > 0 else 0.0

    fg_correct = int(mask_c.sum())
    fg_error = int(mask_e.sum())
    fg_union = int(union_mask.sum())
    total_px = int(H * W)

    overlay = union_img.astype(np.float32)
    ov = overlay[missing_mask]
    overlay[missing_mask] = 0.7 * np.array([255, 0, 0], np.float32) + 0.3 * ov
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    save_rgb(out_union, union_img)
    save_mask(out_missing_mask, missing_mask)
    save_rgb(out_missing_overlay, overlay)

    stats = {
        "canvas_width": W,
        "canvas_height": H,
        "total_pixels": total_px,
        "fg_correct": fg_correct,
        "fg_error": fg_error,
        "fg_union": fg_union,
        "target_mask_mode": target_mode,
        "target_pixels": total_target,
        "missing_pixels": missing_px,
        "missing_ratio_percent": missing_ratio,
        "union_output_path": str(out_union),
        "union_missing_mask_path": str(out_missing_mask),
        "union_missing_overlay_path": str(out_missing_overlay),
        "correct_input_path": str(correct_path),
        "error_input_path": str(error_path),
    }

    with open(out_stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    stats["union_stats_json_path"] = str(out_stats_json)
    return stats


# =========================
# 主流程
# =========================

def build_runtime_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    vars_cfg = raw_cfg.get("vars", {})
    formatted = format_obj(raw_cfg, vars_cfg)

    paths = formatted.get("paths", {})
    options = formatted.get("options", {})
    names = formatted.get("names", {})

    runtime_cfg = {
        **vars_cfg,
        **paths,
        **options,
        **names,
    }
    return runtime_cfg


def main(yaml_obj=None):
    if yaml_obj is None:
        parser = argparse.ArgumentParser(description="打包版：檢驗單輪分割 + 分類是否有像素遺失")
        parser.add_argument("--config", type=str, required=True, help="yaml 設定檔路徑")
        args = parser.parse_args()

        cfg_path = Path(args.config)
        print(f"✅ Using cfg: {cfg_path}")
        raw_cfg = load_yaml(cfg_path)
    else:
        raw_cfg = yaml_obj
    cfg = build_runtime_config(raw_cfg)

    
    for k in ("train_model_method", "map_number", "round"):
        if k in cfg:
            print(f"{k}: {cfg[k]}")

    error_info = reconstruct_error_canvas(cfg)
    print(f"→ 錯誤整合完成：{error_info['error_reconstructed_path']}")

    correct_info = reconstruct_correct_canvas(cfg)
    print(f"→ 正確整合完成：{correct_info['correct_only_path']}")

    union_info = union_correct_and_error(cfg, {**error_info, **correct_info})
    print("=" * 72)
    print(f"畫布尺寸: {union_info['canvas_width']} x {union_info['canvas_height']} (總像素 {union_info['total_pixels']})")
    print(f"正確前景像素: {union_info['fg_correct']}")
    print(f"錯誤前景像素: {union_info['fg_error']}")
    print(f"整合(聯集)前景像素: {union_info['fg_union']}")
    print(f"target mode: {union_info['target_mask_mode']}")
    print(f"有效區像素: {union_info['target_pixels']}")
    print(
        f"有效區內缺失像素: {union_info['missing_pixels']} / {union_info['target_pixels']} "
        f"({union_info['missing_ratio_percent']:.6f}%)"
    )
    print("輸出：")
    print(f"- 錯誤整合圖：{error_info['error_reconstructed_path']}")
    print(f"- 正確整合圖：{correct_info['correct_only_path']}")
    print(f"- 整合圖：{union_info['union_output_path']}")
    print(f"- 缺失遮罩：{union_info['union_missing_mask_path']}")
    print(f"- 缺失標記：{union_info['union_missing_overlay_path']}")
    print(f"- 統計 json：{union_info['union_stats_json_path']}")


if __name__ == "__main__":
    main()