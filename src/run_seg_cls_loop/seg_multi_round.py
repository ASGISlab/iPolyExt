# -*- coding: utf-8 -*-
"""
多輪分割：
- 讀取上一輪分類錯誤圖（ERR_ROOT/*.png）
- 對每張 mid 做 SAM 分割
- 輸出每個 tile 的 pairs / mid
- 最後每個 tile 額外補一張 residual（若存在）
"""
import argparse
import yaml
from pathlib import Path
import re
import os
import gc
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import hashlib
import pickle
def compute_file_hash(file_path, algorithm='sha256'):
    """Compute the hash of a file using the specified algorithm."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as file:
        # Read the file in chunks of 8192 bytes
        while chunk := file.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()
#from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import ray
from ray.util.actor_pool import ActorPool
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
    
    def infer(self, pad, mp):
        masks = self.mask_gen.generate(pad)
        return masks, mp

@ray.remote
def generate_mid(masks_fg, idx, H, W, y0, x0, img_rgb, mid_dir, src_tag, Htile, Wtile):
    imgs_nonwhite={}
    covered = np.zeros((Htile, Wtile), dtype=bool)
    for mm in masks_fg:
        seg_crop = mm["segmentation"]
        h_c, w_c = seg_crop.shape

        # mid 尺寸輸出
        seg_full_mid = np.zeros((H, W), dtype=bool)
        seg_full_mid[y0:y0 + h_c, x0:x0 + w_c] = seg_crop

        masked = np.full_like(img_rgb, 255)
        masked[seg_full_mid] = img_rgb[seg_full_mid]

        cv2.imwrite(
            str(mid_dir / f"mid_{src_tag}_{idx:03d}.png"),
            cv2.cvtColor(masked, cv2.COLOR_RGB2BGR)
        )
        imgs_nonwhite[str(mid_dir / f"mid_{src_tag}_{idx:03d}.png")] = (~is_white_mask(masked)).sum()

        # tile 尺寸統計（覆蓋聯集）：同樣左上對齊貼回 tile 畫布
        seg_full_tile = np.zeros((Htile, Wtile), dtype=bool)
        seg_full_tile[y0:y0 + h_c, x0:x0 + w_c] = seg_crop
        covered |= seg_full_tile
        idx+=1
    return covered, imgs_nonwhite

# ════════════════════════════════════════════════════════════════════
# 0. 讀取 YAML 設定
# ════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config file."
    )
    return parser.parse_args()


def load_yaml_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def is_white_mask(img_np: np.ndarray) -> np.ndarray:
    """嚴格白判定：三通道皆等於 WHITE_BGR 才算白。"""
    b, g, r = (255, 255, 255)
    return (
        (img_np[..., 0] == b) &
        (img_np[..., 1] == g) &
        (img_np[..., 2] == r)
    )

def main(yaml_obj=None):
    if yaml_obj is None:
        args = parse_args()
        cfg = load_yaml_config(args.config)
    else:
        cfg = yaml_obj

    # ---------------------------
    # A. 基本任務參數
    # ---------------------------
    TRAIN_MODEL_METHOD = cfg["task"]["train_model_method"]
    MAP_NUMBER = str(cfg["task"]["map_number"])
    ROUND = int(cfg["task"]["round"])
    num_workers = int(cfg["task"]["num_workers"])

    # ---------------------------
    # B. 路徑參數
    # ---------------------------
    ERR_ROOT_TEMPLATE = cfg["paths"]["err_root_template"]
    OUT_ROOT_TEMPLATE = cfg["paths"]["out_root_template"]
    SAM_CKPT = cfg["paths"]["sam_ckpt"]

    # ---------------------------
    # C. 裝置 / 模型參數
    # ---------------------------
    DEVICE = cfg["model"]["device"]
    if DEVICE == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_TYPE = cfg["model"]["model_type"]

    # ---------------------------
    # D. SAM 生成器參數
    # ---------------------------
    SAM_POINTS_PER_SIDE = int(cfg["sam_generator"]["points_per_side"])
    SAM_POINTS_PER_BATCH = int(cfg["sam_generator"]["points_per_batch"])
    SAM_PRED_IOU_THRESH = float(cfg["sam_generator"]["pred_iou_thresh"])
    SAM_STABILITY_SCORE_THRESH = float(cfg["sam_generator"]["stability_score_thresh"])
    SAM_CROP_N_LAYERS = int(cfg["sam_generator"]["crop_n_layers"])
    SAM_CROP_N_POINTS_DOWNSCALE_FACTOR = int(cfg["sam_generator"]["crop_n_points_downscale_factor"])
    SAM_OUTPUT_MODE = cfg["sam_generator"]["output_mode"]
    gpu_ids = list(set(cfg["sam_generator"]["gpu_ids"].split(',')))
    num_gpus = len(gpu_ids)
    sam_procs = int(cfg["sam_generator"]["sam_procs"])
    assert all([int(i)>=0 for i in gpu_ids])
    assert int(cfg["sam_generator"]["sam_procs"])>=1
    # ---------------------------
    # E. 流程控制參數
    # ---------------------------
    BBOX_PAD = int(cfg["runtime"]["bbox_pad"])
    OVERLAY_ALPHA = float(cfg["runtime"]["overlay_alpha"])
    RESIDUAL_ALPHA = float(cfg["runtime"]["residual_alpha"])
    JPEG_QUALITY = int(cfg["runtime"]["jpeg_quality"])
    SAVE_FALLBACK_ORIGINAL_MID = bool(cfg["runtime"]["save_fallback_original_mid"])
    SAVE_TILE_RESIDUAL = bool(cfg["runtime"]["save_tile_residual"])

    # ---------------------------
    # F. 命名 / Regex 參數
    # ---------------------------
    FNAME_PATTERN = cfg["regex"]["fname_pattern"]
    MID_STEM_PATTERN = cfg["regex"]["mid_stem_pattern"]
    SANITIZE_PATTERN = cfg["regex"]["sanitize_pattern"]


    # ════════════════════════════════════════════════════════════════════
    # 1. 路徑展開
    # ════════════════════════════════════════════════════════════════════
    PREV_ROUND = ROUND - 1

    ERR_ROOT = Path(
        ERR_ROOT_TEMPLATE.format(
            train_model_method=TRAIN_MODEL_METHOD,
            map_number=MAP_NUMBER,
            prev_round=PREV_ROUND,
        )
    )

    OUT_ROOT = Path(
        OUT_ROOT_TEMPLATE.format(
            train_model_method=TRAIN_MODEL_METHOD,
            map_number=MAP_NUMBER,
            round=ROUND,
        )
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


    # ════════════════════════════════════════════════════════════════════
    # 2. SAM 初始化
    # ════════════════════════════════════════════════════════════════════
    #sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CKPT).to(DEVICE)

    #mask_gen_ = SamAutomaticMaskGenerator(
        #sam,
        #points_per_side=SAM_POINTS_PER_SIDE,
        #points_per_batch=SAM_POINTS_PER_BATCH,
        #pred_iou_thresh=SAM_PRED_IOU_THRESH,
        #stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        #crop_n_layers=SAM_CROP_N_LAYERS,
        #crop_n_points_downscale_factor=SAM_CROP_N_POINTS_DOWNSCALE_FACTOR,
        #output_mode=SAM_OUTPUT_MODE,
    #)
    workers = [sam_worker.remote(gpu_ids[i%num_gpus] ,MODEL_TYPE, SAM_CKPT, DEVICE, SAM_POINTS_PER_BATCH, SAM_CROP_N_LAYERS, SAM_CROP_N_POINTS_DOWNSCALE_FACTOR, SAM_PRED_IOU_THRESH, SAM_STABILITY_SCORE_THRESH) 
                for i in range(num_gpus*sam_procs)]
    mask_gen = ActorPool(workers)

    # ════════════════════════════════════════════════════════════════════
    # 3. 小工具
    # ════════════════════════════════════════════════════════════════════
    def is_white_mask_rgb(rgb: np.ndarray) -> np.ndarray:
        return (rgb[..., 0] == 255) & (rgb[..., 1] == 255) & (rgb[..., 2] == 255)


    def residual_bbox(nonwhite: np.ndarray, pad: int, H: int, W: int):
        ys, xs = np.where(nonwhite)
        if ys.size == 0:
            return None

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        y0 = max(0, y0 - pad)
        y1 = min(H, y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(W, x1 + pad)
        return y0, y1, x0, x1


    def sanitize_tag(s: str) -> str:
        return re.sub(SANITIZE_PATTERN, "_", s)


    def ensure_tile_state(tile_state: dict, key, H: int, W: int):
        """
        若首次出現即建立；
        若之後遇到更大尺寸就向右下擴張（保留已累積內容）。
        """
        if key not in tile_state:
            tile_state[key] = {
                "H": H,
                "W": W,
                "union": np.zeros((H, W), dtype=bool),
                "covered": np.zeros((H, W), dtype=bool),
                "base": np.full((H, W, 3), 255, np.uint8),
            }
        else:
            st = tile_state[key]
            if H > st["H"] or W > st["W"]:
                Hnew, Wnew = max(H, st["H"]), max(W, st["W"])

                u2 = np.zeros((Hnew, Wnew), dtype=bool)
                u2[:st["H"], :st["W"]] = st["union"]
                st["union"] = u2

                c2 = np.zeros((Hnew, Wnew), dtype=bool)
                c2[:st["H"], :st["W"]] = st["covered"]
                st["covered"] = c2

                b2 = np.full((Hnew, Wnew, 3), 255, np.uint8)
                b2[:st["H"], :st["W"]] = st["base"]
                st["base"] = b2

                st["H"], st["W"] = Hnew, Wnew


    # ════════════════════════════════════════════════════════════════════
    # 4. 主流程：處理所有錯誤 mid (*.png)
    # ════════════════════════════════════════════════════════════════════
    fname_pat = re.compile(FNAME_PATTERN, re.IGNORECASE)
    mid_files = sorted(ERR_ROOT.glob("*.png"))

    # 逐 tile 累積狀態：非白聯集、已覆蓋聯集、與基底像素（維持左上對齊）
    tile_state = {}  # (ty, tx) -> {"H":Htile,"W":Wtile,"union":bool2d,"covered":bool2d,"base":rgb}
    submitted=0
    for mp in mid_files:
        m = fname_pat.search(mp.stem)
        if not m:
            continue
        img_bgr = cv2.imread(str(mp))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        nonwhite_full = ~is_white_mask_rgb(img_rgb)
        if nonwhite_full.sum() > 0:
            bbox = residual_bbox(nonwhite_full, BBOX_PAD, H, W)
            if bbox is None:
                bbox = (0, H, 0, W)
            y0, y1, x0, x1 = bbox
            crop_rgb = img_rgb[y0:y1, x0:x1]
            mask_gen.submit(lambda a, v: a.infer.remote(*v), (crop_rgb, mp))
            submitted+=1
    imgs_nonwhite={}
    
    pbar = tqdm(range(submitted), desc=f"SAM-round{ROUND} mids")
    for _ in range(submitted):
        masks, mp = mask_gen.get_next_unordered()
        m = fname_pat.search(mp.stem)

        ty, tx = map(int, m.groups())
        key = (ty, tx)

        # 來源唯一 tag（避免覆蓋，保留 _ty/_tx，並加 8 位雜湊）
        m_tag = re.match(MID_STEM_PATTERN, mp.stem)  # 把最外層 _001 去掉
        base_tag = m_tag.group(1) if m_tag else mp.stem
        digest = hashlib.md5(mp.stem.encode()).hexdigest()[:8]
        src_tag = sanitize_tag(f"{base_tag}_{digest}")

        img_bgr = cv2.imread(str(mp))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # 準備 / 擴張此 tile 的累積畫布
        ensure_tile_state(tile_state, key, H, W)
        st = tile_state[key]
        Htile, Wtile = st["H"], st["W"]

        nonwhite_full = ~is_white_mask_rgb(img_rgb)

        # 累積到 tile 的「非白聯集」與基底像素（左上對齊）
        st["union"][:H, :W] |= nonwhite_full
        base_patch = st["base"][:H, :W]
        base_patch[nonwhite_full] = img_rgb[nonwhite_full]

        tile_dir = OUT_ROOT / f"tile_{ty}_{tx}"
        mid_dir = tile_dir / "mid"
        mid_dir.mkdir(parents=True, exist_ok=True)
        saved = 0

        if nonwhite_full.sum() > 0:
            bbox = residual_bbox(nonwhite_full, BBOX_PAD, H, W)
            if bbox is None:
                bbox = (0, H, 0, W)

            y0, y1, x0, x1 = bbox
            crop_rgb = img_rgb[y0:y1, x0:x1]
            nonwhite_crop = nonwhite_full[y0:y1, x0:x1]

            #masks_ = mask_gen_.generate(crop_rgb)
            #masks = mask_gen.get_next()
            #assert all([np.array_equal(v1,v2) if isinstance(v1, np.ndarray) else v1==v2 for i1,i2 in zip(masks_, masks) for v1,v2 in zip(list(i1.values()), list(i2.values()))])
            masks_fg = []

            for mm in masks:
                seg = mm["segmentation"]
                if seg is None:
                    continue
                if seg.dtype != bool:
                    seg = seg.astype(bool)

                # 只保留與非白有交集；為避免吃到白區，限縮成交集
                inter = seg & nonwhite_crop
                if inter.sum() > 0:
                    masks_fg.append({"segmentation": inter})

            # 與首輪一致：大 -> 小，降低 1-px 縫
            masks_fg.sort(key=lambda x: x["segmentation"].sum(), reverse=True)

            if masks_fg:
                masks_fg_parts = [list(item) for item in np.array_split(masks_fg, num_workers)]
                union_mask = []
                for part in masks_fg_parts:
                    tmp=[]
                    for fm in part:
                        seg_crop = fm["segmentation"]
                        h_c, w_c = seg_crop.shape
                        seg_full_mid = np.zeros((H, W), dtype=bool)
                        seg_full_mid[y0:y0 + h_c, x0:x0 + w_c] = seg_crop
                        tmp.append(seg_full_mid)
                    union_mask.append(np.logical_or.reduce(tmp))
                for i in range(1, len(union_mask)):
                    union_mask[i]=np.logical_or(union_mask[i], union_mask[i-1])
                union_mask = [np.zeros_like(union_mask[0])] + union_mask[:-1]
                init_idx = np.cumsum(np.insert([len(item) for item in masks_fg_parts], 0, 0))[:-1]+1
                future_ids = [
                    generate_mid.remote(chunk, init_idx[i], H, W, y0, x0, img_rgb, mid_dir, src_tag, Htile, Wtile) 
                    for i, chunk in enumerate(masks_fg_parts) if len(chunk) > 0
                ]
                partial_results = ray.get(future_ids)
                for part in partial_results:
                    covered, imgs_nonwhite_ = part
                    saved+=1
                    imgs_nonwhite|=imgs_nonwhite_
                    st["covered"] |= covered

        if saved == 0 and SAVE_FALLBACK_ORIGINAL_MID:
            # 沒任何 seg，保底輸出原 mid 一張
            cv2.imwrite(
                str(mid_dir / f"mid_{src_tag}_001.png"),
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            )

        pbar.update(1)
        torch.cuda.empty_cache()
        gc.collect()
    assert not mask_gen.has_next(), 'SHIT, there shoudnt be any jobs left in SAM workers...'
    pbar.close()
    with open(OUT_ROOT/'imgs_nonwhite.pkl', 'wb') as f:
        pickle.dump(imgs_nonwhite, f)


    # ════════════════════════════════════════════════════════════════════
    # 5. 逐 tile 輸出整體 residual（若存在）
    # ════════════════════════════════════════════════════════════════════
    if SAVE_TILE_RESIDUAL:
        for (ty, tx), st in tile_state.items():
            residual_mask = st["union"] & (~st["covered"])
            if not residual_mask.any():
                continue

            mid_dir = OUT_ROOT / f"tile_{ty}_{tx}" / "mid"
            mid_dir.mkdir(parents=True, exist_ok=True)

            # mid：白底 + 原像素（尺寸 = 此 tile 下遇到的最大 H×W）
            residual_mid = np.full((st["H"], st["W"], 3), 255, np.uint8)
            residual_mid[residual_mask] = st["base"][residual_mask]
            cv2.imwrite(
                str(mid_dir / "mid_residual.png"),
                cv2.cvtColor(residual_mid, cv2.COLOR_RGB2BGR)
            )

    print(f"✅ SAM round {ROUND} 完成！所有輸出 → {OUT_ROOT}")