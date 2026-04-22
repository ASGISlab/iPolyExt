# -*- coding: utf-8 -*-
"""
SAM 只跑「前處理圖」。
最後輸出結構：round_x/tile_y_x/{pairs,mid}/  (不產生 tile_parts_all / tile_crops_all)
"""

import os
import cv2
import csv
import gc
import math
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm   # 純文字，不走 ipywidgets
import argparse
import yaml
import ray
from ray.util.actor_pool import ActorPool
import pickle
import hashlib


# ── 0. 讀取 yaml 參數 ───────────────────────────────────
def load_yaml_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_file_hash(file_path, algorithm='sha256'):
    """Compute the hash of a file using the specified algorithm."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as file:
        # Read the file in chunks of 8192 bytes
        while chunk := file.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()
@ray.remote(max_restarts=0, max_task_retries=0, num_cpus=1.0)
class sam_worker:
    def __init__(self, gpu_id,model_typ, sam_ckpt, device, sam_points_per_batch, sam_crop_n_layers, sam_crop_n_points_downscale_factor, sam_pred_iou_thresh, sam_stability_score_thresh):
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[model_typ](checkpoint=sam_ckpt).to(device)
        self.mask_gen = SamAutomaticMaskGenerator(
            sam,
            points_per_batch=sam_points_per_batch,
            crop_n_layers=sam_crop_n_layers,
            crop_n_points_downscale_factor=sam_crop_n_points_downscale_factor,
            pred_iou_thresh=sam_pred_iou_thresh,
            stability_score_thresh=sam_stability_score_thresh,
        )
    
    def infer(self, pad, y0, x0):
        masks = self.mask_gen.generate(pad)
        return masks, y0, x0

def is_white_mask(img_np: np.ndarray) -> np.ndarray:
    """嚴格白判定：三通道皆等於 WHITE_BGR 才算白。"""
    b, g, r = (255, 255, 255)
    return (
        (img_np[..., 0] == b) &
        (img_np[..., 1] == g) &
        (img_np[..., 2] == r)
    )

@ray.remote
def generate_mid(idx, mid, final_masks, x0, y0, tile_img, mid_dir, initial_mask):
    rows_csv=[]
    imgs_nonwhite={}
    counter = 0
    for fm in final_masks:
        mask_for_recon = np.logical_or(fm["mask"], initial_mask)
        mask = fm["mask"]
        ys, xs = np.where(mask_for_recon)
        ymi, yma, xmi, xma = ys.min(), ys.max(), xs.min(), xs.max()

        # CSV
        rows_csv.append([
            mid + counter,
            fm["area"],
            f"{fm['pred_iou']:.4f}",
            f"{fm['stab']:.4f}",
            y0,
            x0,
            xmi,
            ymi,
            xma - xmi + 1,
            yma - ymi + 1
        ])


        masked = np.full_like(tile_img, 255)
        masked[mask] = tile_img[mask]

        # mid
        cv2.imwrite(
            str(mid_dir / f"mid_{idx:03d}.png"),
            cv2.cvtColor(masked, cv2.COLOR_RGB2BGR)
        )
        assert mask.sum()==(~is_white_mask(masked)).sum()
        imgs_nonwhite[str(mid_dir / f"mid_{idx:03d}.png")] = (~is_white_mask(masked)).sum()

        counter += 1
        idx+=1
    return imgs_nonwhite, rows_csv, counter

def main(yaml_obj=None):
    if yaml_obj is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True, help="yaml config path")
        args = parser.parse_args()

        cfg = load_yaml_config(args.config)
    else:
        cfg = yaml_obj

    # ===== 基本資料 =====
    train_model_method = cfg["basic"]["train_model_method"]
    Map_number = str(cfg["basic"]["Map_number"])
    ROUND = int(cfg["basic"]["ROUND"])
    num_workers = int(cfg["basic"]["num_workers"])

    # ===== 路徑 =====
    img_path = cfg["paths"]["img_path"].format(
        train_model_method=train_model_method,
        Map_number=Map_number,
        ROUND=ROUND,
    )
    sam_ckpt = cfg["paths"]["sam_ckpt"]
    model_typ = cfg["paths"]["model_typ"]

    round_root = Path(
        cfg["paths"]["round_root"].format(
            train_model_method=train_model_method,
            Map_number=Map_number,
            ROUND=ROUND,
        )
    )
    csv_filename = cfg["paths"]["csv_filename"]
    full_concat_filename = cfg["paths"]["full_concat_filename"]

    # ===== 裝置 =====
    device_cfg = cfg["device"]["device"]
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    # ===== Tile 設定 =====
    tile_size = int(cfg["tile"]["tile_size"])
    pad_value = int(cfg["tile"]["pad_value"])

    # ===== SAM 參數 =====
    sam_points_per_batch = int(cfg["sam"]["sam_points_per_batch"])
    sam_crop_n_layers = int(cfg["sam"]["sam_crop_n_layers"])
    sam_crop_n_points_downscale_factor = int(cfg["sam"]["sam_crop_n_points_downscale_factor"])
    sam_pred_iou_thresh = float(cfg["sam"]["sam_pred_iou_thresh"])
    sam_stability_score_thresh = float(cfg["sam"]["sam_stability_score_thresh"])
    gpu_ids = list(set(cfg["sam"]["gpu_ids"].split(',')))
    num_gpus = len(gpu_ids)
    sam_procs = int(cfg["sam"]["sam_procs"])
    assert all([int(i)>=0 for i in gpu_ids])
    assert int(cfg["sam"]["sam_procs"])>=1

    # ===== 輸出控制 =====
    save_full_concat = bool(cfg["output"]["save_full_concat"])
    save_demo_jpg = bool(cfg["output"]["save_demo_jpg"])
    demo_jpg_quality = int(cfg["output"]["demo_jpg_quality"])
    overlay_alpha = float(cfg["output"]["overlay_alpha"])
    append_background_mask = bool(cfg["output"]["append_background_mask"])

    # ===== 顏色控制 =====
    random_seed = cfg["color"]["random_seed"]
    if random_seed is not None:
        np.random.seed(int(random_seed))

    # ── 1. 建立輸出資料夾 ─────────────────────────────────────
    round_root.mkdir(parents=True, exist_ok=True)
    csv_path = str(round_root / csv_filename)


    # ── 2. 載入單一張圖 ─────────────────────────────────────
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"讀不到影像：{img_path}")

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    print(f"➡  輸入圖大小：{W}×{H}")


    # ── 3. SAM 初始化 ─────────────────────────────────────
    workers = [sam_worker.remote(gpu_ids[i%num_gpus] ,model_typ, sam_ckpt, device, sam_points_per_batch, sam_crop_n_layers, sam_crop_n_points_downscale_factor, sam_pred_iou_thresh, sam_stability_score_thresh) 
               for i in range(num_gpus*sam_procs)]
    mask_gen = ActorPool(workers)
    #sam = sam_model_registry[model_typ](checkpoint=sam_ckpt).to(device)
    #mask_gen = SamAutomaticMaskGenerator(
        #sam,
        #points_per_batch=sam_points_per_batch,
        #crop_n_layers=sam_crop_n_layers,
        #crop_n_points_downscale_factor=sam_crop_n_points_downscale_factor,
        #pred_iou_thresh=sam_pred_iou_thresh,
        #stability_score_thresh=sam_stability_score_thresh,
    #)


    # ── 4. tile → SAM → 輸出 ───────────────────────────────
    """
    SAM 只讀取單一張圖；SAM 與所有輸出 (pair/mid) 都使用同一張圖像素。
    最後輸出結構：round_x/tile_y_x/{pairs,mid}/  (不產生 tile_parts_all / tile_crops_all)
    """
    header = [
        "id", "area", "pred_iou", "stab", "tile_row", "tile_col",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h"
    ]
    rows_csv = []
    mid = 1

    total_tiles = math.ceil(H / tile_size) * math.ceil(W / tile_size)
    pbar = tqdm(total=total_tiles, desc="Tiles")

    for y0 in range(0, H, tile_size):
        for x0 in range(0, W, tile_size):
            y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)

            tile_img = img[y0:y1, x0:x1]
            real_h, real_w = tile_img.shape[:2]

            # padding 到固定 tile_size
            pad = np.full((tile_size, tile_size, 3), pad_value, dtype=np.uint8)
            pad[:real_h, :real_w] = tile_img

            mask_gen.submit(lambda a, v: a.infer.remote(*v), (pad, y0, x0))
    imgs_nonwhite={}
    for _ in range(0, H, tile_size):
        for _ in range(0, W, tile_size):

            masks, y0, x0 = mask_gen.get_next_unordered()
            y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)

            tile_img = img[y0:y1, x0:x1]
            real_h, real_w = tile_img.shape[:2]

            # padding 到固定 tile_size
            pad = np.full((tile_size, tile_size, 3), pad_value, dtype=np.uint8)
            pad[:real_h, :real_w] = tile_img
            ms = [m for m in masks if m["segmentation"][:real_h, :real_w].sum() > 0]
            ms.sort(key=lambda m: m["segmentation"].sum(), reverse=True)

            # 每個 tile 都建立 mid / pairs
            tile_name = f"tile_{y0}_{x0}"
            tile_dir = round_root / tile_name
            pair_dir = tile_dir / "pairs"
            mid_dir = tile_dir / "mid"
            pair_dir.mkdir(parents=True, exist_ok=True)
            mid_dir.mkdir(parents=True, exist_ok=True)

            # 沒有任何 mask：只留 demo(tile_x_y.jpg)；mid/ 已建立但為空
            if not ms:
                del masks, ms
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                pbar.update(1)
                continue

            # 最終遮罩按小 → 大
            final_masks = [
                {
                    "mask": m["segmentation"][:real_h, :real_w].astype(bool),
                    "area": int(m["segmentation"][:real_h, :real_w].sum()),
                    "pred_iou": m["predicted_iou"],
                    "stab": m["stability_score"],
                }
                for m in ms
            ]
            final_masks.sort(key=lambda d: d["area"])

            # 追加 tile 殘餘區域
            if append_background_mask:
                union_mask = np.zeros((real_h, real_w), dtype=bool)
                for fm in final_masks:
                    union_mask |= fm["mask"]

                bg_mask = ~union_mask
                bg_area = int(bg_mask.sum())
                if bg_area > 0:
                    final_masks.append({
                        "mask": bg_mask,
                        "area": bg_area,
                        "pred_iou": 1.00,
                        "stab": 1.00
                    })

                final_masks.sort(key=lambda d: d["area"])

            final_masks_parts = [list(item) for item in np.array_split(final_masks, num_workers)]
            union_mask = [np.logical_or.reduce([fm['mask'] for fm in part]) for part in final_masks_parts]
            for i in range(1, len(union_mask)):
                union_mask[i]=np.logical_or(union_mask[i], union_mask[i-1])
            union_mask = [np.zeros_like(union_mask[0])] + union_mask[:-1]
            init_idx = np.cumsum(np.insert([len(item) for item in final_masks_parts], 0, 0))[:-1]+1
            future_ids = [
                generate_mid.remote(init_idx[i], mid, chunk, x0, y0, tile_img, mid_dir, union_mask[i]) 
                for i, chunk in enumerate(final_masks_parts) if len(chunk) > 0
            ]
            partial_results = ray.get(future_ids)
            for part in partial_results:
                imgs_nonwhite_, rows_csv_, counter = part
                mid+=counter
                imgs_nonwhite|=imgs_nonwhite_
                rows_csv.extend(rows_csv_)

            del masks, ms, final_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            pbar.update(1)

    pbar.close()
    with open(round_root/'imgs_nonwhite.pkl', 'wb') as f:
        pickle.dump(imgs_nonwhite, f)


    # ── 5. CSV & 全圖對照 ───────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows_csv)

    print("🟢 全部輸出已完成；每個 tile 僅輸出 pairs/ 與 mid/（不產生 tile_crops_all / tile_parts_all）。")
if __name__ == "__main__":
    main()
