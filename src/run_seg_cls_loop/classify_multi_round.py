# -*- coding: utf-8 -*-
"""
多輪分類（整併上一輪 merged，不覆蓋既有正確像素）：
- 讀取上一輪 merged/{core}_full.png 建立 confirmed_any 與初始畫布
- 本輪僅在「未被確認」區域內進行貼圖
- 錯誤 mid：一律與目前 confirmed_any 比較，取 residual = 非白且未覆蓋 的殘餘部分；
  - residual > 0  → 以「原 mid 尺寸」輸出錯誤圖（非 residual 處設白），errors.csv: reason=...|trim_saved
  - residual == 0 → 不輸出圖，errors.csv: reason=...|trim_empty
- fully_covered → 記錄到 skipped.csv，covered_by=prev_round_canvas
- 全白 → skipped.csv: reason=all_white
- 有效像素 < HALF_AREA（未覆蓋區）→ 直接收進 composite_small，不單獨輸出

本版目的：
- 只做「參數整合」
- 盡量不改你原本核心邏輯
- 方便後續拆成 py + yaml
"""

import re
import csv
import pickle, math, rasterio
import gc
import shutil
import imagesize
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set, Optional
import argparse
from cmaas_utils.io import loadLayoutJson, loadLegendJson

import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import timm
import yaml
import ray
import os
import hashlib
from ray.util.actor_pool import ActorPool
# =========================================================
# 0) YAML 載入 + 統一參數區
# =========================================================
def compute_file_hash(file_path, algorithm='sha256'):
    """Compute the hash of a file using the specified algorithm."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as file:
        # Read the file in chunks of 8192 bytes
        while chunk := file.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def load_yaml_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@ray.remote
def load_img(mid_list):
    img_mem={}
    for mp in mid_list:
        img = cv2.imread(str(mp))
        img_mem[str(mp)]=img
    return img_mem

@ray.remote
def write_img(write_buff):
    for p, img in write_buff:
        cv2.imwrite(p, img)

@ray.remote(max_restarts=0, max_task_retries=0, num_cpus=1.0)
class cls_worker:
    def __init__(self, gpu_id, CKPT_PATH, MODEL_NAME, PATCH_H, PATCH_W, MEAN, STD, BATCH_SIZE, cfg, i, num_workers):
        self.id=i
        num_preload = int(cfg["experiment"].get("num_preload"))
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
        Map_number = str(cfg["experiment"]["map_number"])
        ROUND = int(cfg["experiment"]["round"])

        train_model_method = str(cfg["experiment"]["train_model_method"])
        method = str(cfg["experiment"]["method"])
        data_name = str(cfg["experiment"]["data_name"])

        MAP_NAME = str(cfg["experiment"].get("map_name", Map_number))

        # -------------------------
        # B. 路徑主根目錄
        # -------------------------
        PROJECT_ROOT = Path(cfg["paths"]["project_root"])
        LEGACY_CELL_ROOT = Path(
            cfg["paths"].get(
                "legacy_cell_root",
                str(PROJECT_ROOT / "過去的SAM_分類循環jupyter_cell")
            )
        )

        # -------------------------
        # C. 模型參數
        # -------------------------
        MODEL_NAME = str(cfg["model"]["model_name"])
        BATCH_SIZE = int(cfg["model"]["batch_size"])
        CONF_TH = float(cfg["model"]["conf_th"])

        PATCH_H = int(cfg["model"]["patch_h"])
        PATCH_W = int(cfg["model"]["patch_w"])
        STRIDE_H = int(cfg["model"].get("stride_h", PATCH_H))
        STRIDE_W = int(cfg["model"].get("stride_w", PATCH_W))

        MEAN = list(cfg["model"]["mean"])
        STD = list(cfg["model"]["std"])

        # -------------------------
        # D. 多數決 / 分支規則參數
        # -------------------------
        MAJORITY_TH = float(cfg["vote"]["majority_th"])
        MIN_PATCH = int(cfg["vote"]["min_patch"])
        MIN_NONBLANK = int(cfg["vote"]["min_nonblank"])
        STRONG_MIN = int(cfg["vote"]["strong_min"])
        
        blank_white_area_ratio = float(cfg["rules"]["blank_white_area_ratio"])
        small_threshold_extra_cols = int(cfg["rules"]["small_threshold_extra_cols"])
        # -------------------------
        # E. 小圖 / 殘餘圖規則
        # -------------------------
        #HALF_AREA = int(cfg["rules"].get("half_area", (PATCH_H * PATCH_W) // 2))

        #small_threshold = cfg["rules"].get("small_threshold", None)
        #if small_threshold is None:
        #    extra_cols = int(cfg["rules"].get("small_threshold_extra_cols", 60))
        #    SMALL_THRESHOLD = PATCH_H * PATCH_W + PATCH_H * extra_cols
        #else:
        #    SMALL_THRESHOLD = int(small_threshold)
        HALF_AREA = int(round(PATCH_H * PATCH_W* blank_white_area_ratio)) 
        SMALL_THRESHOLD = int(round(PATCH_H * PATCH_W + PATCH_H*small_threshold_extra_cols))
        # -------------------------
        # F. 標籤相關
        # -------------------------
        BLANK_NAME = str(cfg["rules"].get("blank_name", "blank"))

        # -------------------------
        # G. 是否保留 debug print
        # -------------------------
        ENABLE_DEBUG_PRINT = bool(cfg["rules"].get("enable_debug_print", True))


        # =========================================================
        # 1) 由統一參數推導出的路徑
        # =========================================================
        RUN_NAME = cfg["paths"]["run_name"]
        CURR_SEG_ROOT = (
            LEGACY_CELL_ROOT
            / RUN_NAME
            / "sam1"
            / train_model_method
            / MAP_NAME
            / f"round_{ROUND}"
        )

        PREV_OUT_ROOT = (
            LEGACY_CELL_ROOT
            / RUN_NAME
            / "classify"
            / "SAM_post_classify_out"
            / train_model_method
            / MAP_NAME
            / f"round_{ROUND - 1}"
            if ROUND > 1 else None
        )

        PREV_MERGED_DIR = PREV_OUT_ROOT / "segmentation_verified" / "merged" if PREV_OUT_ROOT else None

        CURR_OUT_ROOT = (
            LEGACY_CELL_ROOT
            / RUN_NAME
            / "classify"
            / "SAM_post_classify_out"
            / train_model_method
            / MAP_NAME
            / f"round_{ROUND}"
        )

        CORRECT_DIR = CURR_OUT_ROOT / "segmentation_verified"
        MERGED_DIR = CORRECT_DIR / "merged"

        ERROR_DIR_RAW = CURR_OUT_ROOT / "segmentation_rejected__raw"
        ERROR_DIR_FINAL = CURR_OUT_ROOT / "segmentation_rejected"
        # =========================================================
        # 2) 輸出目錄初始化
        # =========================================================
        ERROR_DIR = ERROR_DIR_RAW
        # =========================================================
        # 3) 共用工具
        # =========================================================
        with open(CURR_SEG_ROOT/'imgs_nonwhite.pkl', 'rb') as f:
            imgs_nonwhite = pickle.load(f)  

        def is_white_mask(img_np: np.ndarray) -> np.ndarray:
            """嚴格白：三通道皆 255"""
            return (img_np[..., 0] == 255) & (img_np[..., 1] == 255) & (img_np[..., 2] == 255)


        def canon_label(label: str) -> Optional[str]:
            """
            統一輸出用的 canonical label：
            - 去掉每段的 '_後綴'，保留 '-' 複合結構
            - 例：'Mag-Mk_17_13' -> 'Mag-Mk'
                'Mag_01-Qy_17' -> 'Mag-Qy'
                'Qo_001'       -> 'Qo'
                'blank' / 'blank_xxx' -> 'blank'
            """
            if label is None:
                return None

            s = str(label).strip()
            if not s:
                return None

            head = s.split('_', 1)[0].lower()
            if head.startswith("blank"):
                return BLANK_NAME

            parts = []
            for p in s.split('-'):
                p0 = p.split('_', 1)[0].strip()
                if p0:
                    parts.append(p0)

            if not parts:
                return None

            return "-".join(parts)


        def core_set_from_label(label: str) -> Set[str]:
            """
            回傳『單一 canonical label』的集合：
            - 'Mag-Mk_17_13' -> {'Mag-Mk'}
            - 'Qo_001'       -> {'Qo'}
            - blank          -> 空集合（不參與一致性）
            """
            c = canon_label(label)
            if (not c) or (c == BLANK_NAME):
                return set()
            return {c}


        def choose_core_from_intersection(intersection: Set[str], counts: Counter) -> Optional[str]:
            if not intersection:
                return None
            if len(intersection) == 1:
                return next(iter(intersection))
            return max(intersection, key=lambda k: counts.get(k, 0))


        def overlay_new_pixels(dst: np.ndarray, src: np.ndarray, src_nonW_mask: np.ndarray, y: int, x: int, confirmed_any: np.ndarray):
            """
            僅把 src 的「非白 且 未被確認」像素貼到 dst，並回傳新覆蓋布林遮罩。
            """
            h, w = src.shape[:2]
            patch_dst = dst[y:y + h, x:x + w]
            nonwhite = src_nonW_mask#~is_white_mask(src)
            newmask = nonwhite & (~confirmed_any)
            patch_dst[newmask] = src[newmask]
            return newmask


        def pad_to_shape(arr: np.ndarray, H: int, W: int, fill_value=255) -> np.ndarray:
            """把圖像/遮罩補到目標大小；RGB 用 255（白），布林用 False。"""
            h, w = arr.shape[:2]
            if (h, w) == (H, W):
                return arr

            if arr.ndim == 3:
                out = np.full((H, W, 3), fill_value, arr.dtype)
                out[:h, :w] = arr
                return out

            out = np.zeros((H, W), dtype=arr.dtype)
            out[:h, :w] = arr
            return out


        # =========================================================
        # 4) 載入模型與前處理
        # =========================================================

        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        label_to_idx = ckpt["label_to_idx"]
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        NUM_CLASSES = len(label_to_idx)

        state_dict = ckpt["state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        if "head.weight" in state_dict:
            assert state_dict["head.weight"].shape[0] == NUM_CLASSES, (
                f"head.weight 輸出數 {state_dict['head.weight'].shape[0]} != NUM_CLASSES {NUM_CLASSES}"
            )

        self.model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=NUM_CLASSES,
            img_size=(PATCH_H, PATCH_W),
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.val_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


        def batched_infer(patches: List[Image.Image]) -> Tuple[List[str], List[float]]:
            labels, confs = [], []
            if not patches:
                return labels, confs

            tensors = [self.val_tf(p) for p in patches]
            batch = torch.stack(tensors, dim=0).to(self.device)

            logits_all = []
            with torch.no_grad():
                for i in range(0, len(batch), BATCH_SIZE):
                    b = batch[i:i + BATCH_SIZE]
                    logits_all.append(self.model(b))

            logits = torch.cat(logits_all, dim=0)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)

            conf = conf.detach().cpu().numpy().tolist()
            pred = pred.detach().cpu().numpy().tolist()

            for c, idx in zip(conf, pred):
                labels.append(idx_to_label[idx])
                confs.append(float(c))

            return labels, confs


        def classify_image_for_core(pil_img: Image.Image, is_img_all_white, conf_th: float = CONF_TH) -> Dict:
            """
            新版決策（兩條互斥路：D vs B/C），且 patch 推論只做一次。

            前置：對每個非全白 patch 做推論，得到 pred/conf。
            有效票定義：非全白且 canon_label(pred) != blank 才算票。

            統計：
            total_nonblank  = 有效票數
            strong_nonblank = 有效票中 conf >= CONF_TH 的票數

            決策（互斥）：
            if total_nonblank >= MIN_PATCH:                 -> Branch D（主投票）
            elif total_nonblank >= MIN_NONBLANK and
                strong_nonblank >= STRONG_MIN:             -> Branch B/C（強弱一致性）
            else:                                           -> error（或交給外層小殘餘規則）
            """
            np_img = np.array(pil_img)
            H, W = np_img.shape[:2]

            if is_img_all_white:
                return {
                    "status": "error",
                    "assigned_core": "blank",
                    "strong_cnt": 0,
                    "weak_cnt": 0,
                    "core_counts": {},
                    "reason": "retry_blank_image",
                }

            patches: List[Image.Image] = []
            for y in range(0, H, STRIDE_H):
                for x in range(0, W, STRIDE_W):
                    w_eff = min(PATCH_W, W - x)
                    h_eff = min(PATCH_H, H - y)
                    if w_eff * h_eff <= 0:
                        continue

                    patch = pil_img.crop((x, y, x + w_eff, y + h_eff))

                    if (w_eff, h_eff) != (PATCH_W, PATCH_H):
                        pad = Image.new("RGB", (PATCH_W, PATCH_H), (255, 255, 255))
                        pad.paste(patch, (0, 0))
                        patch = pad

                    arr = np.array(patch)
                    if is_white_mask(arr).all():
                        continue

                    patches.append(patch)

            if not patches:
                return {
                    "status": "correct",
                    "assigned_core": "blank",
                    "strong_cnt": 0,
                    "weak_cnt": 0,
                    "core_counts": {},
                    "reason": "no_nonwhite_patches",
                }

            labels, confs = batched_infer(patches)

            core_counts = Counter()
            strong_sets: List[Set[str]] = []
            weak_sets: List[Set[str]] = []

            blank_pred_cnt = 0
            total_nonblank = 0
            strong_nonblank = 0

            for lab, c in zip(labels, confs):
                s = core_set_from_label(lab)
                if not s:
                    blank_pred_cnt += 1
                    continue

                c_lab = next(iter(s))
                total_nonblank += 1
                core_counts[c_lab] += 1

                if c >= conf_th:
                    strong_nonblank += 1
                    strong_sets.append(s)
                else:
                    weak_sets.append(s)

            weak_nonblank = total_nonblank - strong_nonblank

            if total_nonblank == 0:
                return {
                    "status": "correct",
                    "assigned_core": "blank",
                    "strong_cnt": 0,
                    "weak_cnt": 0,
                    "core_counts": {},
                    "reason": f"all_pred_blank(nonwhite_patches={len(patches)})",
                }

            if total_nonblank >= MIN_PATCH:
                cand, hit = core_counts.most_common(1)[0]
                ratio = hit / total_nonblank
                if ratio >= MAJORITY_TH:
                    return {
                        "status": "correct",
                        "assigned_core": cand,
                        "strong_cnt": strong_nonblank,
                        "weak_cnt": weak_nonblank,
                        "core_counts": dict(core_counts),
                        "reason": f"branch_D_majority_{ratio:.3f}(hit={hit}/{total_nonblank})",
                    }

                return {
                    "status": "error",
                    "assigned_core": cand,
                    "strong_cnt": strong_nonblank,
                    "weak_cnt": weak_nonblank,
                    "core_counts": dict(core_counts),
                    "reason": f"branch_D_fail_{ratio:.3f}(hit={hit}/{total_nonblank})",
                }

            if (total_nonblank >= MIN_NONBLANK) and (strong_nonblank >= STRONG_MIN):
                inter = set.intersection(*strong_sets) if strong_sets else set()
                if inter:
                    cand = choose_core_from_intersection(inter, core_counts)
                    weak_mismatch = sum(1 for s in weak_sets if cand not in s)
                    if weak_mismatch == 0:
                        return {
                            "status": "correct",
                            "assigned_core": cand,
                            "strong_cnt": strong_nonblank,
                            "weak_cnt": weak_nonblank,
                            "core_counts": dict(core_counts),
                            "reason": "branch_BC_ok(strong_intersection)",
                        }

                    return {
                        "status": "error",
                        "assigned_core": cand,
                        "strong_cnt": strong_nonblank,
                        "weak_cnt": weak_nonblank,
                        "core_counts": dict(core_counts),
                        "reason": f"branch_BC_fail_weak_mismatch({weak_mismatch})",
                    }

                all_sets = strong_sets + weak_sets
                inter_all = set.intersection(*all_sets) if all_sets else set()
                if inter_all:
                    cand = choose_core_from_intersection(inter_all, core_counts)
                    weak_mismatch = sum(1 for s in weak_sets if cand not in s)
                    if weak_mismatch == 0:
                        return {
                            "status": "correct",
                            "assigned_core": cand,
                            "strong_cnt": strong_nonblank,
                            "weak_cnt": weak_nonblank,
                            "core_counts": dict(core_counts),
                            "reason": "branch_BC_ok(all_intersection)",
                        }

                    return {
                        "status": "error",
                        "assigned_core": cand,
                        "strong_cnt": strong_nonblank,
                        "weak_cnt": weak_nonblank,
                        "core_counts": dict(core_counts),
                        "reason": f"branch_BC_fail_weak_mismatch_all({weak_mismatch})",
                    }

                return {
                    "status": "error",
                    "assigned_core": None,
                    "strong_cnt": strong_nonblank,
                    "weak_cnt": weak_nonblank,
                    "core_counts": dict(core_counts),
                    "reason": "branch_BC_fail_no_intersection",
                }

            return {
                "status": "error",
                "assigned_core": None,
                "strong_cnt": strong_nonblank,
                "weak_cnt": weak_nonblank,
                "core_counts": dict(core_counts),
                "reason": (
                    f"insufficient_votes(total={total_nonblank}, "
                    f"strong={strong_nonblank}, blank_pred={blank_pred_cnt})"
                ),
            }
        
        def process_tiles(ty, tx, mid_list, Hc_cur, Wc_cur, tile_h, tile_w, pbar):
            from ray.experimental.tqdm_ray import tqdm
            canvases: Dict[str, np.ndarray] = {}
            confirmed_any = np.zeros((Hc_cur, Wc_cur), dtype=bool)
            target_H, target_W = Hc_cur, Wc_cur
            tile_y = slice(ty, ty + tile_h)
            tile_x = slice(tx, tx + tile_w)
            write_buff=[]

            if PREV_MERGED_DIR and PREV_MERGED_DIR.exists():
                prev_cores = sorted([p for p in PREV_MERGED_DIR.glob("*.png")])
                Hprev = Wprev = 0
                tmp_prev = {}
                for p in prev_cores:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue

                    tmp_prev[p.stem.replace("_full", "")] = img
                    Hprev = max(Hprev, img.shape[0])
                    Wprev = max(Wprev, img.shape[1])

                target_H = max(target_H, Hprev)
                target_W = max(target_W, Wprev)

                if confirmed_any.shape != (target_H, target_W):
                    confirmed_any = pad_to_shape(confirmed_any, target_H, target_W)

                for core, img in tmp_prev.items():
                    rgb = pad_to_shape(img, target_H, target_W, fill_value=255)
                    canvases[core] = rgb
                    confirmed_any |= (~is_white_mask(rgb))

            if ENABLE_DEBUG_PRINT:
                print(
                    "[debug] PREV_MERGED_DIR:", PREV_MERGED_DIR,
                    "exists:", PREV_MERGED_DIR.exists() if PREV_MERGED_DIR else None
                )
                print("[debug] prev cores loaded:", list(canvases.keys())[:8], "count=", len(canvases))
                print(
                    "[debug] target canvas size:", (target_H, target_W),
                    "confirmed_any shape:", confirmed_any.shape
                )


            def ensure_canvas(core: str):
                if core not in canvases:
                    canvases[core] = np.full((target_H, target_W, 3), 255, np.uint8)
            err_rows = [[
                "map", "tile_y", "tile_x", "mid_file", "status", "assigned_core",
                "strong_cnt", "weak_cnt", "top_core_counts", "reason"
            ]]
            skip_rows = [[
                "map", "tile_y", "tile_x", "mid_file", "reason", "covered_by"
            ]]
            hw_list = []
            for p in mid_list:#change by kevin: super slow WTF
                #im0 = cv2.imread(str(p))
                w0, h0 = imagesize.get(str(p))
                hw_list.append((h0, w0))
                #if im0 is None:
                    #continue
                #h0, w0 = im0.shape[:2]
                #hw_list.append((h0, w0))

            if not hw_list:
                return

            max_h = max(h for h, _ in hw_list)
            max_w = max(w for _, w in hw_list)
            del hw_list

            composite_small = np.full((max_h, max_w, 3), 255, np.uint8)
            pending_errors = []
            
            chunks = [mid_list[i : i + num_preload] for i in range(0, len(mid_list), num_preload)]
            img_mem={}
            unresolved_mid_list = []
            for mp in mid_list:
                pbar.update.remote(1)
                if str(mp) not in img_mem:
                    mid_list_parts = [list(item) for item in np.array_split(chunks[0], num_workers)]
                    future = [
                        load_img.remote(chunk) 
                        for i, chunk in enumerate(mid_list_parts) if len(chunk) > 0
                    ]
                    img_mem.clear()
                    for img_mem_part in ray.get(future):
                        img_mem |= img_mem_part
                    del future
                    chunks = chunks[1:]
                img = img_mem[str(mp)]
                if img is None:
                    tmp = img_mem.pop(str(mp))
                    continue

                h, w = img.shape[:2]
                real_ty, real_tx = ty, tx

                if confirmed_any.shape[0] < real_ty + h or confirmed_any.shape[1] < real_tx + w:
                    newH = max(confirmed_any.shape[0], real_ty + h, target_H)
                    newW = max(confirmed_any.shape[1], real_tx + w, target_W)

                    confirmed_any = pad_to_shape(
                        confirmed_any.astype(np.uint8), newH, newW, fill_value=0
                    ).astype(bool)

                    for k in list(canvases.keys()):
                        canvases[k] = pad_to_shape(canvases[k], newH, newW, fill_value=255)

                    target_H, target_W = newH, newW

                sl_y = slice(real_ty, real_ty + h)
                sl_x = slice(real_tx, real_tx + w)

                nonwhite = ~is_white_mask(img)
                if str(mp) in imgs_nonwhite:
                    nonwhite_num = imgs_nonwhite[str(mp)]
                else:
                    nonwhite_num =nonwhite.sum()
                if nonwhite_num == 0:
                    #skip_rows.append([MAP_NAME, real_ty, real_tx, mp.name, "all_white", ""])
                    tmp = img_mem.pop(str(mp))
                    del tmp      
                    continue

                already = confirmed_any[sl_y, sl_x]
                uncovered = nonwhite & (~already)
                uncovered_sum= uncovered.sum()
                if uncovered_sum == 0:
                    #skip_rows.append([
                        #MAP_NAME, real_ty, real_tx, mp.name,
                        #"fully_covered", "prev_round_canvas"
                    #])
                    tmp = img_mem.pop(str(mp))
                    del tmp      
                    continue

                if uncovered_sum < HALF_AREA:
                    patch = composite_small[0:h, 0:w]
                    patch[uncovered] = img[uncovered]
                    tmp = img_mem.pop(str(mp))
                    del tmp      
                    continue

                img_for_cls = img.copy()
                img_for_cls[already] = 255
                pil = Image.fromarray(cv2.cvtColor(img_for_cls, cv2.COLOR_BGR2RGB))
                res = classify_image_for_core(pil, False, conf_th=CONF_TH)#pil, (~nonwhite).all(), conf_th=CONF_TH

                if res["status"] == "correct":
                    core = res["assigned_core"]
                    if core and core != BLANK_NAME:
                        ensure_canvas(core)
                        overlay_new_pixels(canvases[core], img, nonwhite, real_ty, real_tx, already)
                        already |= nonwhite
                else:
                    cc = Counter(res["core_counts"])
                    top = "; ".join(f"{k}:{v}" for k, v in cc.most_common(5))
                    unresolved_mid_list.append(mp)
                    pending_errors.append({
                        "mp": mp,
                        "real_ty": real_ty,
                        "real_tx": real_tx,
                        "h": h,
                        "w": w,
                        "status": res["status"],
                        "assigned": (res["assigned_core"] or ""),
                        "strong_cnt": res["strong_cnt"],
                        "weak_cnt": res["weak_cnt"],
                        "top": top,
                        "reason_raw": res["reason"],
                    })
                tmp = img_mem.pop(str(mp))
                del tmp
                torch.cuda.empty_cache()
                gc.collect()

            if (~is_white_mask(composite_small)).any():
                dst_name = f"{MAP_NAME}_ty{ty}_tx{tx}_composite_small.png"
                write_buff.append((str(ERROR_DIR / dst_name), composite_small.copy()))
                if len(write_buff)>=num_preload:
                    chunk_size = math.ceil(len(write_buff) / num_workers)
                    write_buff_parts = [
                        write_buff[i : i + chunk_size] 
                        for i in range(0, len(write_buff), chunk_size)
                    ]
                    future = [
                        write_img.remote(chunk) 
                        for i, chunk in enumerate(write_buff_parts) if len(chunk) > 0
                    ]
                    ray.get(future)
                    write_buff=[]
                #cv2.imwrite(str(ERROR_DIR / dst_name), composite_small)
                #err_rows.append([
                    #MAP_NAME, ty, tx, dst_name,
                    #"retry", "", 0, 0, "", "small_residual_collected"
                #])
            chunks = [unresolved_mid_list[i : i + num_preload] for i in range(0, len(unresolved_mid_list), num_preload)]
            for e in pending_errors:
                mp = e["mp"]
                real_ty = e["real_ty"]
                real_tx = e["real_tx"]
                h = e["h"]
                w = e["w"]
                if str(mp) not in img_mem:
                    mid_list_parts = [list(item) for item in np.array_split(chunks[0], num_workers)]
                    future = [
                        load_img.remote(chunk) 
                        for i, chunk in enumerate(mid_list_parts) if len(chunk) > 0
                    ]
                    img_mem.clear()
                    for img_mem_part in ray.get(future):
                        img_mem |= img_mem_part
                    del future
                    chunks = chunks[1:]
                img = img_mem[str(mp)]
                if img is None:
                    #err_rows.append([
                        #MAP_NAME, real_ty, real_tx, mp.name,
                        #e["status"], e["assigned"],
                        #e["strong_cnt"], e["weak_cnt"], e["top"],
                        #e["reason_raw"] + "|io_error"
                    #])
                    continue

                sl_y = slice(real_ty, real_ty + h)
                sl_x = slice(real_tx, real_tx + w)
                nonwhite = ~is_white_mask(img)
                already = confirmed_any[sl_y, sl_x]
                residual = nonwhite & (~already)
                res_pixels = int(residual.sum())

                if res_pixels == 0:
                    #err_rows.append([
                        #MAP_NAME, real_ty, real_tx, mp.name,
                        #"error", e["assigned"],
                        #e["strong_cnt"], e["weak_cnt"], e["top"],
                        #e["reason_raw"] + "|trim_empty"
                    #])
                    continue

                residual_img = np.full_like(img, 255)
                residual_img[residual] = img[residual]

                dst_name = f"{MAP_NAME}_ty{real_ty}_tx{real_tx}_{mp.name}"
                write_buff.append((str(ERROR_DIR / dst_name), residual_img))
                if len(write_buff)>=num_preload:
                    chunk_size = math.ceil(len(write_buff) / num_workers)
                    write_buff_parts = [
                        write_buff[i : i + chunk_size] 
                        for i in range(0, len(write_buff), chunk_size)
                    ]
                    future = [
                        write_img.remote(chunk) 
                        for i, chunk in enumerate(write_buff_parts) if len(chunk) > 0
                    ]
                    ray.get(future)
                    write_buff=[]
                #cv2.imwrite(str(ERROR_DIR / dst_name), residual_img)

                #err_rows.append([
                    #MAP_NAME, real_ty, real_tx, dst_name,
                    #"retry", e["assigned"],
                    #e["strong_cnt"], e["weak_cnt"], e["top"],
                    #e["reason_raw"] + "|trim_saved"
                #])
            if len(write_buff)>0:
                chunk_size = math.ceil(len(write_buff) / num_workers)
                write_buff_parts = [
                    write_buff[i : i + chunk_size] 
                    for i in range(0, len(write_buff), chunk_size)
                ]
                future = [
                    write_img.remote(chunk) 
                    for i, chunk in enumerate(write_buff_parts) if len(chunk) > 0
                ]
                ray.get(future)
            del img_mem
            return confirmed_any[tile_y, tile_x], canvases, err_rows, skip_rows, target_H, target_W
        self.process_tiles = process_tiles
    def infer(self, ty, tx, mid_list, Hc, Wc, tile_h, tile_w, pbar):
        res, canv, err, skip, tH, tW =self.process_tiles(ty, tx, mid_list, Hc, Wc, tile_h, tile_w, pbar)
        return ty, tx, res, canv, err, skip, tH, tW

def main(yaml_obj=None):
    if yaml_obj is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True, help="yaml config path")
        args = parser.parse_args()

        cfg = load_yaml_config(args.config)
    else:
        cfg = yaml_obj

    # -------------------------
    # A. 基本實驗資訊
    # -------------------------
    Map_number = str(cfg["experiment"]["map_number"])
    ROUND = int(cfg["experiment"]["round"])

    train_model_method = str(cfg["experiment"]["train_model_method"])
    method = str(cfg["experiment"]["method"])
    data_name = str(cfg["experiment"]["data_name"])

    MAP_NAME = str(cfg["experiment"].get("map_name", Map_number))
    map_dir = str(cfg["experiment"].get("map_dir"))

    num_workers = int(cfg["experiment"].get("num_workers"))
    underscore_replace = str(cfg["experiment"].get("underscore_replace"))

    # -------------------------
    # B. 路徑主根目錄
    # -------------------------
    PROJECT_ROOT = Path(cfg["paths"]["project_root"])
    LEGACY_CELL_ROOT = Path(
        cfg["paths"].get(
            "legacy_cell_root",
            str(PROJECT_ROOT / "過去的SAM_分類循環jupyter_cell")
        )
    )

    # -------------------------
    # C. 模型參數
    # -------------------------
    MODEL_NAME = str(cfg["model"]["model_name"])
    BATCH_SIZE = int(cfg["model"]["batch_size"])
    CONF_TH = float(cfg["model"]["conf_th"])

    gpu_ids = list(set(cfg["model"]["gpu_ids"].split(',')))
    num_gpus = len(gpu_ids)
    cls_procs = int(cfg["model"]["cls_procs"])

    PATCH_H = int(cfg["model"]["patch_h"])
    PATCH_W = int(cfg["model"]["patch_w"])
    STRIDE_H = int(cfg["model"].get("stride_h", PATCH_H))
    STRIDE_W = int(cfg["model"].get("stride_w", PATCH_W))

    MEAN = list(cfg["model"]["mean"])
    STD = list(cfg["model"]["std"])

    # -------------------------
    # D. 多數決 / 分支規則參數
    # -------------------------
    MAJORITY_TH = float(cfg["vote"]["majority_th"])
    MIN_PATCH = int(cfg["vote"]["min_patch"])
    MIN_NONBLANK = int(cfg["vote"]["min_nonblank"])
    STRONG_MIN = int(cfg["vote"]["strong_min"])

    # -------------------------
    # E. 小圖 / 殘餘圖規則
    # -------------------------
    HALF_AREA = int(cfg["rules"].get("half_area", (PATCH_H * PATCH_W) // 2))

    small_threshold = cfg["rules"].get("small_threshold", None)
    if small_threshold is None:
        extra_cols = int(cfg["rules"].get("small_threshold_extra_cols", 60))
        SMALL_THRESHOLD = PATCH_H * PATCH_W + PATCH_H * extra_cols
    else:
        SMALL_THRESHOLD = int(small_threshold)
    # -------------------------
    # F. 標籤相關
    # -------------------------
    BLANK_NAME = str(cfg["rules"].get("blank_name", "blank"))

    # -------------------------
    # G. 是否保留 debug print
    # -------------------------
    ENABLE_DEBUG_PRINT = bool(cfg["rules"].get("enable_debug_print", True))


    # =========================================================
    # 1) 由統一參數推導出的路徑
    # =========================================================
    RUN_NAME = cfg["paths"]["run_name"]
    CURR_SEG_ROOT = (
        LEGACY_CELL_ROOT
        / RUN_NAME
        / "sam1"
        / train_model_method
        / MAP_NAME
        / f"round_{ROUND}"
    )

    PREV_OUT_ROOT = (
        LEGACY_CELL_ROOT
        / RUN_NAME
        / "classify"
        / "SAM_post_classify_out"
        / train_model_method
        / MAP_NAME
        / f"round_{ROUND - 1}"
        if ROUND > 1 else None
    )

    PREV_MERGED_DIR = PREV_OUT_ROOT / "segmentation_verified" / "merged" if PREV_OUT_ROOT else None

    CURR_OUT_ROOT = (
        LEGACY_CELL_ROOT
        / RUN_NAME
        / "classify"
        / "SAM_post_classify_out"
        / train_model_method
        / MAP_NAME
        / f"round_{ROUND}"
    )

    CORRECT_DIR = CURR_OUT_ROOT / "segmentation_verified"
    MERGED_DIR = CORRECT_DIR / "merged"
    TIF_OUTPUT_DIR = Path(cfg["paths"]["tif_output_dir"])    

    ERROR_DIR_RAW = CURR_OUT_ROOT / "segmentation_rejected__raw"
    ERROR_DIR_FINAL = CURR_OUT_ROOT / "segmentation_rejected"

    CKPT_PATH = str(cfg["paths"]["ckpt_path"])


    # =========================================================
    # 2) 輸出目錄初始化
    # =========================================================

    if ERROR_DIR_RAW.exists():
        shutil.rmtree(ERROR_DIR_RAW)
    ERROR_DIR_RAW.mkdir(parents=True, exist_ok=True)

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    TIF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURR_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ERROR_DIR = ERROR_DIR_RAW

    # =========================================================
    # 3) 共用工具
    # =========================================================

    def save_trimmed_error_image(img: np.ndarray, residual_mask: np.ndarray, out_path: Path):
        """
        保留原 mid 尺寸，非 residual 的像素設白（255）；residual 處保留原像素。
        img: BGR (cv2 讀取)
        residual_mask: bool, 與 img 同高寬
        """
        out = np.full_like(img, 255)
        out[residual_mask] = img[residual_mask]
        cv2.imwrite(str(out_path), out)


    def is_white_mask(img_np: np.ndarray) -> np.ndarray:
        """嚴格白：三通道皆 255"""
        return (img_np[..., 0] == 255) & (img_np[..., 1] == 255) & (img_np[..., 2] == 255)


    def canon_label(label: str) -> Optional[str]:
        """
        統一輸出用的 canonical label：
        - 去掉每段的 '_後綴'，保留 '-' 複合結構
        - 例：'Mag-Mk_17_13' -> 'Mag-Mk'
            'Mag_01-Qy_17' -> 'Mag-Qy'
            'Qo_001'       -> 'Qo'
            'blank' / 'blank_xxx' -> 'blank'
        """
        if label is None:
            return None

        s = str(label).strip()
        if not s:
            return None

        head = s.split('_', 1)[0].lower()
        if head.startswith("blank"):
            return BLANK_NAME

        parts = []
        for p in s.split('-'):
            p0 = p.split('_', 1)[0].strip()
            if p0:
                parts.append(p0)

        if not parts:
            return None

        return "-".join(parts)


    def core_set_from_label(label: str) -> Set[str]:
        """
        回傳『單一 canonical label』的集合：
        - 'Mag-Mk_17_13' -> {'Mag-Mk'}
        - 'Qo_001'       -> {'Qo'}
        - blank          -> 空集合（不參與一致性）
        """
        c = canon_label(label)
        if (not c) or (c == BLANK_NAME):
            return set()
        return {c}


    def choose_core_from_intersection(intersection: Set[str], counts: Counter) -> Optional[str]:
        if not intersection:
            return None
        if len(intersection) == 1:
            return next(iter(intersection))
        return max(intersection, key=lambda k: counts.get(k, 0))


    def overlay_new_pixels(dst: np.ndarray, src: np.ndarray, y: int, x: int, confirmed_any: np.ndarray):
        """
        僅把 src 的「非白 且 未被確認」像素貼到 dst，並回傳新覆蓋布林遮罩。
        """
        h, w = src.shape[:2]
        patch_dst = dst[y:y + h, x:x + w]
        patch_conf = confirmed_any[y:y + h, x:x + w]
        nonwhite = ~is_white_mask(src)
        newmask = nonwhite & (~patch_conf)
        patch_dst[newmask] = src[newmask]


    def pad_to_shape(arr: np.ndarray, H: int, W: int, fill_value=255) -> np.ndarray:
        """把圖像/遮罩補到目標大小；RGB 用 255（白），布林用 False。"""
        h, w = arr.shape[:2]
        if (h, w) == (H, W):
            return arr

        if arr.ndim == 3:
            out = np.full((H, W, 3), fill_value, arr.dtype)
            out[:h, :w] = arr
            return out

        out = np.zeros((H, W), dtype=arr.dtype)
        out[:h, :w] = arr
        return out


    # =========================================================
    # 5) 掃描本輪 mids 並估算畫布大小
    # =========================================================

    tile_pat = re.compile(r"tile_([-]?\d+)_([-]?\d+)$")


    def iter_tiles_and_mids(root: Path, imgs_nonwhite):
        """yield ty, tx, [mid_paths]，並按非白像素數由大到小排序"""
        for tdir in sorted(root.glob("tile_*_*")):
            m = tile_pat.match(tdir.name)
            if not m:
                continue

            ty, tx = int(m.group(1)), int(m.group(2))
            mid_dir = tdir / "mid"
            if not mid_dir.exists():
                continue

            mids = list(sorted(mid_dir.glob("*.png")))
            if not mids:
                continue

            areas = []
            for p in mids:
                if str(p) in imgs_nonwhite:
                    nonwhite = imgs_nonwhite[str(p)]
                    areas.append((int(nonwhite), p))
                else:
                    img = cv2.imread(str(p))
                    if img is None:
                        areas.append((0, p))
                    else:
                        nonwhite = (~is_white_mask(img)).sum()
                        areas.append((int(nonwhite), p))

            areas.sort(key=lambda x: x[0], reverse=True)
            yield ty, tx, [p for _, p in areas]


    def estimate_canvas_from_curr_round(root: Path) -> Tuple[int, int, int, int]:
        Hmax = Wmax = 0
        tile_h = tile_w = None

        for ty, tx, mids in root:
            if not mids:
                continue
            w, h = imagesize.get(str(mids[0]))
            #img = cv2.imread(str(mids[0]))
            #if img is None:
                #continue

            #h, w = img.shape[:2]
            if tile_h is None:
                tile_h = h
            if tile_w is None:
                tile_w = w

            Hmax = max(Hmax, ty + h)
            Wmax = max(Wmax, tx + w)

        return Hmax, Wmax, (tile_h or 0), (tile_w or 0)


    def dominant_prev_core(
        canvases: Dict[str, np.ndarray],
        sl_y: slice,
        sl_x: slice,
        nonwhite_mid: np.ndarray,
    ) -> Optional[str]:
        best_core, best = None, 0
        for core, canvas in canvases.items():
            patch = canvas[sl_y, sl_x]
            overlap = (~is_white_mask(patch)) & nonwhite_mid
            cnt = int(overlap.sum())
            if cnt > best:
                best, best_core = cnt, core
        return best_core

    with open(CURR_SEG_ROOT/'imgs_nonwhite.pkl', 'rb') as f:
        imgs_nonwhite = pickle.load(f)       
    tiles_list = list(iter_tiles_and_mids(CURR_SEG_ROOT, imgs_nonwhite))
    Hc_cur, Wc_cur, tile_h, tile_w = estimate_canvas_from_curr_round(tiles_list)
    if Hc_cur == 0 or Wc_cur == 0:
        raise FileNotFoundError(f"[ERROR] 找不到任何 mid：{CURR_SEG_ROOT}")
    # =========================================================
    # 4) 載入模型與前處理
    # =========================================================
    workers = [cls_worker.remote(gpu_ids[i%num_gpus] ,CKPT_PATH, MODEL_NAME, PATCH_H, PATCH_W, MEAN, STD, BATCH_SIZE, cfg, i, num_workers) 
               for i in range(num_gpus*cls_procs)]
    models = ActorPool(workers)

    # =========================================================
    # 6) 準備上一輪 merged 畫布與 confirmed_any
    # =========================================================

    canvases: Dict[str, np.ndarray] = {}
    confirmed_any = np.zeros((Hc_cur, Wc_cur), dtype=bool)
    target_H, target_W = Hc_cur, Wc_cur

    if PREV_MERGED_DIR and PREV_MERGED_DIR.exists():
        prev_cores = sorted([p for p in PREV_MERGED_DIR.glob("*.png")])
        Hprev = Wprev = 0
        tmp_prev = {}

        for p in prev_cores:
            img = cv2.imread(str(p))
            if img is None:
                continue

            tmp_prev[p.stem.replace("_full", "")] = img
            Hprev = max(Hprev, img.shape[0])
            Wprev = max(Wprev, img.shape[1])

        target_H = max(target_H, Hprev)
        target_W = max(target_W, Wprev)

        if confirmed_any.shape != (target_H, target_W):
            confirmed_any = pad_to_shape(confirmed_any, target_H, target_W)

        for core, img in tmp_prev.items():
            rgb = pad_to_shape(img, target_H, target_W, fill_value=255)
            canvases[core] = rgb
            confirmed_any |= (~is_white_mask(rgb))

    if ENABLE_DEBUG_PRINT:
        print(
            "[debug] PREV_MERGED_DIR:", PREV_MERGED_DIR,
            "exists:", PREV_MERGED_DIR.exists() if PREV_MERGED_DIR else None
        )
        print("[debug] prev cores loaded:", list(canvases.keys())[:8], "count=", len(canvases))
        print(
            "[debug] target canvas size:", (target_H, target_W),
            "confirmed_any shape:", confirmed_any.shape
        )


    def ensure_canvas(core: str, H, W):
        if core not in canvases:
            canvases[core] = np.full((H, W, 3), 255, np.uint8)

    # =========================================================
    # 7) 準備 CSV
    # =========================================================

    err_rows = [[
        "map", "tile_y", "tile_x", "mid_file", "status", "assigned_core",
        "strong_cnt", "weak_cnt", "top_core_counts", "reason"
    ]]
    skip_rows = [[
        "map", "tile_y", "tile_x", "mid_file", "reason", "covered_by"
    ]]


    # =========================================================
    # 8) 主流程
    # =========================================================
    total_mids = sum(len(mids) for _, _, mids in tiles_list)
    from ray.experimental import tqdm_ray
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    pbar = remote_tqdm.remote(total=total_mids, desc=f"{MAP_NAME} round{ROUND} tiles")
    for ty, tx, mid_list in tiles_list:
        tile_y = slice(ty, ty + tile_h)
        tile_x = slice(tx, tx + tile_w)
        models.submit(lambda a, v: a.infer.remote(*v), (ty, tx, mid_list, Hc_cur, Wc_cur, tile_h, tile_w, pbar))

    for _ in tiles_list:
        ty, tx, confirmed_any_tile, canvases_tile, err_rows, skip_rows, hhh, www = models.get_next_unordered()
        tile_y = slice(ty, ty + tile_h)
        tile_x = slice(tx, tx + tile_w)
        confirmed_any[tile_y, tile_x] = confirmed_any_tile
        for k,v in canvases_tile.items():
            ensure_canvas(k, hhh, www)
            canvases[k][tile_y, tile_x] = v[tile_y, tile_x]
    pbar.close.remote()

    # =========================================================
    # 9) 輸出 merged / CSV / confirmed_any
    # =========================================================

    for core, canvas in canvases.items():
        with open(f"{map_dir}/{MAP_NAME}.offset", 'r') as f:
            h, w, offset_x, offset_y = map(int, f.read().split())
        poly_pred = ~is_white_mask(canvas)
        mask=np.zeros((h, w), dtype=bool)
        mask[offset_y:offset_y+poly_pred.shape[0], offset_x:offset_x+poly_pred.shape[1]] = poly_pred
        with rasterio.open(TIF_OUTPUT_DIR / f"{MAP_NAME}_{core.replace(underscore_replace, '_')}_poly.tif", 'w', driver='GTiff', compress='lzw', height=h, width=w,count=1, dtype=np.uint8) as fh:
            fh.write(mask[None,:,:].astype(np.uint8))
        out_path = MERGED_DIR / f"{core.replace(underscore_replace, '_')}_full.png"       
        cv2.imwrite(str(out_path), canvas)

    with open(ERROR_DIR / "errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(err_rows)

    with open(ERROR_DIR / "skipped.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(skip_rows)

    np.save(CURR_OUT_ROOT / "confirmed_any.npy", confirmed_any.astype(np.uint8))

    print(f"\n✅ round{ROUND} 完成")
    print(f"  → 合併輸出：{MERGED_DIR}")
    print(f"  → 主體暫存錯誤：{ERROR_DIR_RAW}")


    # =========================================================
    # 10) 小圖整合（post-merge）
    # =========================================================

    SRC_ERROR_DIR = ERROR_DIR_RAW
    DST_ERROR_DIR = ERROR_DIR_FINAL

    SRC_ERROR_CSV = SRC_ERROR_DIR / "errors.csv"
    SRC_SKIPPED_CSV = SRC_ERROR_DIR / "skipped.csv"

    DST_ERROR_CSV = DST_ERROR_DIR / "errors.csv"
    DST_SKIPPED_CSV = DST_ERROR_DIR / "skipped.csv"

    if DST_ERROR_DIR.exists():
        shutil.rmtree(DST_ERROR_DIR)
    DST_ERROR_DIR.mkdir(parents=True, exist_ok=True)

    if SRC_SKIPPED_CSV.exists():
        shutil.copy2(SRC_SKIPPED_CSV, DST_SKIPPED_CSV)

    tile_pat_post = re.compile(r"(?:^|_)ty(-?\d+)_tx(-?\d+)(?:_|$)")

    header = [
        "map", "tile_y", "tile_x", "mid_file", "status", "assigned_core",
        "strong_cnt", "weak_cnt", "top_core_counts", "reason"
    ]
    rows = []

    if SRC_ERROR_CSV.exists():
        with open(SRC_ERROR_CSV, "r", newline="", encoding="utf-8-sig") as f:
            reader = list(csv.reader(f))

        if reader:
            if reader[0] and reader[0][0] == "map":
                header = reader[0]
                rows = reader[1:]
            else:
                rows = reader

    idx = {k: i for i, k in enumerate(header)}


    def append_reason(r, extra):
        if len(r) < len(header):
            r += [""] * (len(header) - len(r))
        r[idx["reason"]] = (r[idx["reason"]] or "") + extra


    for p in SRC_ERROR_DIR.glob("*_composite_small.png"):
        shutil.copy2(p, DST_ERROR_DIR / p.name)

        m = tile_pat_post.search(p.name)
        if not m:
            continue

        ty, tx = int(m.group(1)), int(m.group(2))
        comp_name = p.name

        if not any(len(r) >= len(header) and r[idx["mid_file"]] == comp_name for r in rows):
            new_row = [""] * len(header)
            new_row[idx["map"]] = MAP_NAME
            new_row[idx["tile_y"]] = str(ty)
            new_row[idx["tile_x"]] = str(tx)
            new_row[idx["mid_file"]] = comp_name
            new_row[idx["status"]] = "retry"
            new_row[idx["assigned_core"]] = ""
            new_row[idx["strong_cnt"]] = "0"
            new_row[idx["weak_cnt"]] = "0"
            new_row[idx["top_core_counts"]] = ""
            new_row[idx["reason"]] = "postmerge_copied_existing"
            rows.append(new_row)

    pngs = sorted(SRC_ERROR_DIR.glob("*.png"))

    small_by_tile = defaultdict(list)
    large_to_copy = []

    base_comp_by_tile = {}
    for p in DST_ERROR_DIR.glob("*_composite_small.png"):
        m = tile_pat_post.search(p.name)
        if m:
            base_comp_by_tile[(int(m.group(1)), int(m.group(2)))] = p

    for p in SRC_ERROR_DIR.glob("*_composite_small.png"):
        m = tile_pat_post.search(p.name)
        if m and (int(m.group(1)), int(m.group(2))) not in base_comp_by_tile:
            base_comp_by_tile[(int(m.group(1)), int(m.group(2)))] = p

    for p in tqdm(pngs, desc="Scan & split small/large"):
        if p.name.endswith("_composite_small.png"):
            continue

        m = tile_pat_post.search(p.name)
        if not m:
            continue

        img = cv2.imread(str(p))
        if img is None:
            continue

        nonwhite = int((~is_white_mask(img)).sum())
        if nonwhite < SMALL_THRESHOLD:
            small_by_tile[(int(m.group(1)), int(m.group(2)))].append(p)
        else:
            large_to_copy.append(p)

    for p in tqdm(large_to_copy, desc="Copy large images"):
        shutil.copy2(p, DST_ERROR_DIR / p.name)

    merged_count = 0
    added_tiles = 0
    updated_tiles = 0

    for (ty, tx), file_list in tqdm(list(small_by_tile.items()), desc="Merge small images by tile"):
        base_path = base_comp_by_tile.get((ty, tx))

        sizes = []
        for p in file_list:
            im = cv2.imread(str(p))
            if im is not None:
                sizes.append(im.shape[:2])

        if base_path and base_path.exists():
            imc = cv2.imread(str(base_path))
            if imc is not None:
                sizes.append(imc.shape[:2])

        if not sizes:
            continue

        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)

        if base_path and base_path.exists():
            base_img = cv2.imread(str(base_path))
            if base_img is None:
                base_img = np.full((max_h, max_w, 3), 255, np.uint8)
            else:
                H, W = base_img.shape[:2]
                if (H, W) != (max_h, max_w):
                    padded = np.full((max_h, max_w, 3), 255, base_img.dtype)
                    padded[0:H, 0:W] = base_img
                    base_img = padded
        else:
            base_img = np.full((max_h, max_w, 3), 255, np.uint8)

        for p in file_list:
            img = cv2.imread(str(p))
            if img is None:
                continue

            h, w = img.shape[:2]
            patch_base = base_img[0:h, 0:w]
            nonwhite = ~is_white_mask(img)
            missing = is_white_mask(patch_base) & nonwhite
            if missing.any():
                patch_base[missing] = img[missing]
                merged_count += 1

        comp_name = f"{MAP_NAME}_ty{ty}_tx{tx}_composite_small.png"
        comp_path = DST_ERROR_DIR / comp_name
        cv2.imwrite(str(comp_path), base_img)

        row_found = None
        for r in rows:
            if len(r) >= len(header) and r[idx["mid_file"]] == comp_name:
                row_found = r
                break

        if row_found:
            append_reason(row_found, "|postmerge_added")
            updated_tiles += 1
        else:
            new_row = [""] * len(header)
            new_row[idx["map"]] = MAP_NAME
            new_row[idx["tile_y"]] = str(ty)
            new_row[idx["tile_x"]] = str(tx)
            new_row[idx["mid_file"]] = comp_name
            new_row[idx["status"]] = "retry"
            new_row[idx["assigned_core"]] = ""
            new_row[idx["strong_cnt"]] = "0"
            new_row[idx["weak_cnt"]] = "0"
            new_row[idx["top_core_counts"]] = ""
            sources = ";".join([p.name for p in file_list])
            new_row[idx["reason"]] = f"postmerge_small|sources={sources}"
            rows.append(new_row)
            added_tiles += 1

        for p in file_list:
            for r in rows:
                if len(r) >= len(header) and r[idx["mid_file"]] == p.name:
                    append_reason(r, f"|postmerge_collapsed_into={comp_name}")
                    break

    with open(DST_ERROR_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("[postmerge] 完成。")
    print(f"  - 大圖（>= 門檻）已複製：{len(large_to_copy)} 張")
    print(f"  - 合併小圖：貼入次數 {merged_count}；新增 tile：{added_tiles}；更新 tile：{updated_tiles}")
    print(f"  - 正式輸出：{DST_ERROR_DIR}")

