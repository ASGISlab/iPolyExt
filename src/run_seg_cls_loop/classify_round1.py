# -*- coding: utf-8 -*-
"""
分類第一輪（參數統一版）

目的：
1) 保持原本邏輯不變
2) 把之後要搬去 yaml 的參數，盡量全部集中到最上方
3) 方便後續拆成 py + yaml

目前未改：
- 主流程邏輯
- 投票邏輯
- postmerge 邏輯
"""

import os
import re
import csv, imagesize, rasterio
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import timm
import argparse
import yaml, pickle, math
import ray
from ray.util.actor_pool import ActorPool
# =========================================================
# 0) YAML 載入區（取代原本全部參數區）
# =========================================================

import hashlib
def compute_file_hash(file_path, algorithm='sha256'):
    """Compute the hash of a file using the specified algorithm."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as file:
        # Read the file in chunks of 8192 bytes
        while chunk := file.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML 格式錯誤：{config_path}")
    return cfg

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
    def __init__(self, gpu_id, CKPT_PATH, MODEL_NAME, PATCH_H, PATCH_W, MEAN, STD, BATCH_SIZE, cfg, i):
        self.id=i
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
        task_cfg = cfg["task"]
        path_cfg = cfg["paths"]
        model_cfg = cfg["model"]
        norm_cfg = cfg["normalize"]
        vote_cfg = cfg["vote"]
        blank_cfg = cfg["blank_rule"]
        name_cfg = cfg["names"]
        regex_cfg = cfg["regex"]
        csv_cfg = cfg["csv"]

        # -------------------------
        # A. 任務識別 / 版本資訊
        # -------------------------
        MAP_NUMBER = str(task_cfg["map_number"])
        
        ROUND = int(task_cfg["round"])

        DATA_NAME = str(task_cfg["data_name"])
        METHOD = str(task_cfg["method"])
        TRAIN_MODEL_METHOD = str(task_cfg["train_model_method"])
        num_workers = int(task_cfg.get("num_workers"))
        num_preload = int(task_cfg.get("num_preload"))

        # -------------------------
        # B. 輸入 / 輸出路徑
        # -------------------------
        IN_ROOTS = list(path_cfg["in_roots"])
        OUTPUT_ROOT = str(path_cfg["output_root"])
        CKPT_PATH = str(path_cfg["ckpt_path"])

        # -------------------------
        # C. 模型 / 推論參數
        # -------------------------
        MODEL_NAME = str(model_cfg["model_name"])

        PATCH_H = int(model_cfg["patch_h"])
        PATCH_W = int(model_cfg["patch_w"])
        STRIDE_H = int(model_cfg["stride_h"])
        STRIDE_W = int(model_cfg["stride_w"])

        BATCH_SIZE = int(model_cfg["batch_size"])
        CONF_TH = float(model_cfg["conf_th"])

        gpu_ids = list(set(cfg["model"]["gpu_ids"].split(',')))
        num_gpus = len(gpu_ids)
        cls_procs = int(cfg["model"]["cls_procs"])

        USE_CUDA_IF_AVAILABLE = bool(model_cfg["use_cuda_if_available"])

        # -------------------------
        # D. 正規化參數
        # -------------------------
        MEAN = list(norm_cfg["mean"])
        STD = list(norm_cfg["std"])

        # -------------------------
        # E. 投票 / 決策參數
        # -------------------------
        MAJORITY_TH = float(vote_cfg["majority_th"])
        MIN_PATCH = int(vote_cfg["min_patch"])

        MIN_NONBLANK = int(vote_cfg["min_nonblank"])
        STRONG_MIN = int(vote_cfg["strong_min"])

        # -------------------------
        # F. 白底 / blank / 面積規則
        # -------------------------
        WHITE_BGR = tuple(blank_cfg["white_bgr"])
        BLANK_NAME = str(blank_cfg["blank_name"])

        
        blank_white_area_ratio = float(blank_cfg["blank_white_area_ratio"])
        SMALL_THRESHOLD_EXTRA_W = int(blank_cfg["small_threshold_extra_w"])

        #HALF_AREA = (PATCH_H * PATCH_W) // 2
        #SMALL_THRESHOLD = PATCH_H * PATCH_W + PATCH_H * SMALL_THRESHOLD_EXTRA_W

        HALF_AREA = int(round(PATCH_H * PATCH_W* blank_white_area_ratio)) 
        SMALL_THRESHOLD = int(round(PATCH_H * PATCH_W + PATCH_H*SMALL_THRESHOLD_EXTRA_W))

        # -------------------------
        # G. 資料夾名稱 / 檔名規則
        # -------------------------
        DIRNAME_MID = str(name_cfg["dirname_mid"])

        DIRNAME_CORRECT = str(name_cfg["dirname_correct"])
        DIRNAME_MERGED = str(name_cfg["dirname_merged"])
        DIRNAME_ERROR = str(name_cfg["dirname_error"])
        DIRNAME_ERROR_POSTMERGE_TMP = str(name_cfg["dirname_error_postmerge_tmp"])

        ERROR_CSV_NAME = str(name_cfg["error_csv_name"])
        SKIPPED_CSV_NAME = str(name_cfg["skipped_csv_name"])

        MERGED_CANVAS_FILENAME_TEMPLATE = str(name_cfg["merged_canvas_filename_template"])
        ERROR_IMAGE_FILENAME_TEMPLATE = str(name_cfg["error_image_filename_template"])
        COMPOSITE_SMALL_FILENAME_TEMPLATE = str(name_cfg["composite_small_filename_template"])
        SOURCE_DESC_TEMPLATE = str(name_cfg["source_desc_template"])

        # -------------------------
        # H. tile / regex 規則
        # -------------------------
        TILE_GLOB_PATTERN = str(regex_cfg["tile_glob_pattern"])
        TILE_DIR_REGEX = str(regex_cfg["tile_dir_regex"])
        TILE_COORD_REGEX = str(regex_cfg["tile_coord_regex"])

        # -------------------------
        # I. CSV 欄位
        # -------------------------
        ERROR_CSV_HEADER = list(csv_cfg["error_csv_header"])
        SKIPPED_CSV_HEADER = list(csv_cfg["skipped_csv_header"])

        # =========================================================
        # 1) 工具函式
        # =========================================================

        def is_white_mask(img_np: np.ndarray) -> np.ndarray:
            """嚴格白判定：三通道皆等於 WHITE_BGR 才算白。"""
            b, g, r = WHITE_BGR
            return (
                (img_np[..., 0] == b) &
                (img_np[..., 1] == g) &
                (img_np[..., 2] == r)
            )


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


        def build_canvas_size(tiles_list) -> Tuple[int, int, int, int]:
            """掃描第一張 mid 估算畫布大小與單張 tile H,W"""
            Hmax = Wmax = 0
            tile_h = tile_w = None

            for ty, tx, mids in tiles_list:
                if not mids:
                    continue
                img = cv2.imread(str(mids[0]))
                if img is None:
                    continue

                h, w = img.shape[:2]
                if tile_h is None:
                    tile_h = h
                if tile_w is None:
                    tile_w = w

                Hmax = max(Hmax, ty + h)
                Wmax = max(Wmax, tx + w)

            return Hmax, Wmax, (tile_h or 0), (tile_w or 0)


        def batched_infer(
            patches: List[Image.Image],
            model,
            val_tf,
            idx_to_label,
            device
        ) -> Tuple[List[str], List[float]]:
            labels, confs = [], []
            if not patches:
                return labels, confs

            tensors = [val_tf(p) for p in patches]
            batch = torch.stack(tensors, dim=0).to(device)

            logits_all = []
            with torch.no_grad():
                for i in range(0, len(batch), BATCH_SIZE):
                    b = batch[i:i + BATCH_SIZE]
                    logits_all.append(model(b))

            logits = torch.cat(logits_all, dim=0)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)

            conf = conf.detach().cpu().numpy().tolist()
            pred = pred.detach().cpu().numpy().tolist()

            for c, idx in zip(conf, pred):
                labels.append(idx_to_label[idx])
                confs.append(float(c))

            return labels, confs


        def classify_image_for_core(
            pil_img: Image.Image,
            model,
            val_tf,
            idx_to_label,
            device,
            conf_th: float = CONF_TH
        ) -> Dict:
            """
            新版決策（兩條互斥路：D vs B/C），且 patch 推論只做一次。

            前置：對每個非全白 patch 做推論，得到 pred/conf。
            有效票定義：非全白且 canon_label(pred) != blank 才算票。

            統計：
            total_nonblank  = 有效票數
            strong_nonblank = 有效票中 conf >= CONF_TH 的票數

            決策（互斥）：
            if total_nonblank >= MIN_PATCH:
                -> Branch D（主投票）
            elif total_nonblank >= MIN_NONBLANK and strong_nonblank >= STRONG_MIN:
                -> Branch B/C（強弱一致性）
            else:
                -> error
            """
            np_img = np.array(pil_img)
            H, W = np_img.shape[:2]

            if is_white_mask(np_img).mean() == 1.0:
                return {
                    "status": "error",
                    "assigned_core": BLANK_NAME,
                    "strong_cnt": 0,
                    "weak_cnt": 0,
                    "core_counts": {},
                    "reason": "retry_blank_image"
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
                    "assigned_core": BLANK_NAME,
                    "strong_cnt": 0,
                    "weak_cnt": 0,
                    "core_counts": {},
                    "reason": "no_nonwhite_patches"
                }

            labels, confs = batched_infer(
                patches=patches,
                model=model,
                val_tf=val_tf,
                idx_to_label=idx_to_label,
                device=device
            )

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
                    "assigned_core": BLANK_NAME,
                    "strong_cnt": 0,
                    "weak_cnt": 0,
                    "core_counts": {},
                    "reason": f"all_pred_blank(nonwhite_patches={len(patches)})"
                }

            # Branch D：主投票
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
                        "reason": f"branch_D_majority_{ratio:.3f}(hit={hit}/{total_nonblank})"
                    }
                else:
                    return {
                        "status": "error",
                        "assigned_core": cand,
                        "strong_cnt": strong_nonblank,
                        "weak_cnt": weak_nonblank,
                        "core_counts": dict(core_counts),
                        "reason": f"branch_D_fail_{ratio:.3f}(hit={hit}/{total_nonblank})"
                    }

            # Branch B/C：強弱一致性
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
                            "reason": "branch_BC_ok(strong_intersection)"
                        }
                    else:
                        return {
                            "status": "error",
                            "assigned_core": cand,
                            "strong_cnt": strong_nonblank,
                            "weak_cnt": weak_nonblank,
                            "core_counts": dict(core_counts),
                            "reason": f"branch_BC_fail_weak_mismatch({weak_mismatch})"
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
                            "reason": "branch_BC_ok(all_intersection)"
                        }
                    else:
                        return {
                            "status": "error",
                            "assigned_core": cand,
                            "strong_cnt": strong_nonblank,
                            "weak_cnt": weak_nonblank,
                            "core_counts": dict(core_counts),
                            "reason": f"branch_BC_fail_weak_mismatch_all({weak_mismatch})"
                        }

                return {
                    "status": "error",
                    "assigned_core": None,
                    "strong_cnt": strong_nonblank,
                    "weak_cnt": weak_nonblank,
                    "core_counts": dict(core_counts),
                    "reason": "branch_BC_fail_no_intersection"
                }

            return {
                "status": "error",
                "assigned_core": None,
                "strong_cnt": strong_nonblank,
                "weak_cnt": weak_nonblank,
                "core_counts": dict(core_counts),
                "reason": f"insufficient_votes(total={total_nonblank}, strong={strong_nonblank}, blank_pred={blank_pred_cnt})"
            }


        def overlay_new_pixels(
            dst: np.ndarray,
            src: np.ndarray,
            src_nonW_mask: np.ndarray,
            y: int,
            x: int,
            confirmed_any: np.ndarray
        ):
            """
            僅把 src 的「非白 且 未被確認」像素貼到 dst，
            並回傳這次新覆蓋的布林遮罩（在 dst 的座標系）。
            """
            h, w = src.shape[:2]
            patch_dst = dst[y:y + h, x:x + w]
            patch_conf = confirmed_any
            nonwhite = src_nonW_mask#~is_white_mask(src)
            newmask = nonwhite & (~patch_conf)
            patch_dst[newmask] = src[newmask]
            return newmask


        def append_reason(row, header_idx_map, header_len, extra):
            if len(row) < header_len:
                row += [""] * (header_len - len(row))
            row[header_idx_map["reason"]] = (row[header_idx_map["reason"]] or "") + extra


        # =========================================================
        # 2) 載入模型與轉換
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

        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=NUM_CLASSES,
            img_size=(PATCH_H, PATCH_W),
        )

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        device = torch.device(
            "cuda" if (USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()) else "cpu"
        )
        model.to(device)

        val_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])


        # =========================================================
        # 3) 主流程（多地圖）
        # =========================================================

        OUT = Path(OUTPUT_ROOT)
        tile_coord_re = re.compile(TILE_COORD_REGEX)

        def process_tiles(mid_list, ty, tx, Hc, Wc, map_root, tile_h, tile_w):
            next_src_id = 1
            srcid_to_desc: Dict[int, str] = {}
            method_dir = map_root.parent.parent.name
            group_dir = map_root.parent.name
            out_map_dir = OUT / method_dir / group_dir / f"round_{ROUND}"
            error_dir = out_map_dir / DIRNAME_ERROR
            err_rows = []
            skip_rows = []
            tile_y = slice(ty, ty + tile_h)
            tile_x = slice(tx, tx + tile_w)
            write_buff=[]

            sizes = []
            canvases: Dict[str, np.ndarray] = defaultdict(
                lambda: np.full((Hc, Wc, 3), 255, np.uint8)
            )
            confirmed_any = np.zeros((Hc, Wc), dtype=bool)
            confirmed_id = np.zeros((Hc, Wc), dtype=np.int32)
            for p in mid_list:
                w0, h0 = imagesize.get(str(p))
                sizes.append((h0, w0))
                #img_tmp = cv2.imread(str(p))
                #if img_tmp is not None:
                    #sizes.append(img_tmp.shape[:2])

            if not sizes:
                return

            max_h = max(h for h, _ in sizes)
            max_w = max(w for _, w in sizes)
            composite_small = np.full((max_h, max_w, 3), 255, np.uint8)
            has_small = False

            chunks = [mid_list[i : i + num_preload] for i in range(0, len(mid_list), num_preload)]
            img_mem={}
            for mp in mid_list:
                if str(mp) not in img_mem:
                    mid_list_parts = [list(item) for item in np.array_split(chunks[0], num_workers)]
                    future = [
                        load_img.remote(chunk) 
                        for i, chunk in enumerate(mid_list_parts) if len(chunk) > 0
                    ]
                    for img_mem_part in ray.get(future):
                        img_mem |= img_mem_part
                    del future
                    chunks = chunks[1:]
                img = img_mem[str(mp)]
                if img is None:
                    continue

                h, w = img.shape[:2]
                sl_y = slice(ty, ty + h)
                sl_x = slice(tx, tx + w)

                nonwhite = ~is_white_mask(img)
                if nonwhite.sum() == 0:
                    skip_rows.append([MAP_NUMBER, ty, tx, mp.name, "all_white", ""])
                    tmp = img_mem.pop(str(mp))
                    del tmp
                    continue

                already = confirmed_any[sl_y, sl_x]
                if (nonwhite & already).sum() == nonwhite.sum():
                    cov_ids = confirmed_id[sl_y, sl_x][nonwhite]
                    if cov_ids.size:
                        ids, cnts = np.unique(cov_ids, return_counts=True)
                        top = sorted(zip(ids, cnts), key=lambda x: x[1], reverse=True)[:5]
                        covered_by = "; ".join(f"{srcid_to_desc.get(i, '?')}" for i, _ in top)
                    else:
                        covered_by = ""

                    skip_rows.append([MAP_NUMBER, ty, tx, mp.name, "fully_covered", covered_by])
                    tmp = img_mem.pop(str(mp))
                    del tmp
                    continue

                uncovered = nonwhite & (~already)

                if int(uncovered.sum()) < HALF_AREA:
                    comp_patch = composite_small[0:h, 0:w]
                    comp_patch[uncovered] = img[uncovered]
                    has_small = True
                    tmp = img_mem.pop(str(mp))
                    del tmp                    
                    continue

                img_for_cls = img.copy()
                img_for_cls[already] = 255
                pil = Image.fromarray(cv2.cvtColor(img_for_cls, cv2.COLOR_BGR2RGB))

                res = classify_image_for_core(
                    pil_img=pil,
                    model=model,
                    val_tf=val_tf,
                    idx_to_label=idx_to_label,
                    device=device,
                    conf_th=CONF_TH
                )
                do_rm_img=True
                if res["status"] == "correct":
                    core = res["assigned_core"]
                    if core and core != BLANK_NAME:
                        newmask = overlay_new_pixels(
                            dst=canvases[core],
                            src=img,
                            src_nonW_mask=nonwhite,
                            y=ty,
                            x=tx,
                            confirmed_any=already
                        )

                        already |= newmask
                        if newmask.any():
                            confirmed_id[sl_y, sl_x][newmask] = next_src_id
                            srcid_to_desc[next_src_id] = SOURCE_DESC_TEMPLATE.format(
                                map_name=MAP_NUMBER,
                                round=ROUND,
                                ty=ty,
                                tx=tx,
                                mid_name=mp.name,
                                core=core
                            )
                            next_src_id += 1
                else:
                    dst_name = ERROR_IMAGE_FILENAME_TEMPLATE.format(
                        map_name=MAP_NUMBER,
                        ty=ty,
                        tx=tx,
                        mid_name=mp.name
                    )
                    write_buff.append((str(error_dir / dst_name), np.array(img)))
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
                    #cv2.imwrite(str(error_dir / dst_name), img)

                    cc = Counter(res["core_counts"])
                    top = "; ".join(f"{k}:{v}" for k, v in cc.most_common(5))
                    err_rows.append([
                        MAP_NUMBER, ty, tx, mp.name, res["status"], res["assigned_core"] or "",
                        res["strong_cnt"], res["weak_cnt"], top, res["reason"]
                    ])
                    do_rm_img=False
                if do_rm_img:
                    tmp = img_mem.pop(str(mp))
                    del tmp

            if has_small and (~is_white_mask(composite_small)).any():
                dst_name = COMPOSITE_SMALL_FILENAME_TEMPLATE.format(
                    map_name=MAP_NUMBER,
                    ty=ty,
                    tx=tx
                )
                cv2.imwrite(str(error_dir / dst_name), composite_small)
                err_rows.append([
                    MAP_NUMBER, ty, tx, dst_name, "retry", "", 0, 0, "",
                    "small_residual_collected"
                ])
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
            return confirmed_id[tile_y, tile_x], confirmed_any[tile_y, tile_x], canvases, err_rows, skip_rows
        self.process_tiles = process_tiles
    def infer(self, mid_list, ty, tx, Hc, Wc, map_root, tile_h, tile_w):
        confirmed_id, confirmed_any, canvases, err_rows, skip_rows = self.process_tiles(mid_list, ty, tx, Hc, Wc, map_root, tile_h, tile_w)
        return ty, tx, confirmed_id, confirmed_any, canvases, err_rows, skip_rows
def main(yaml_obj=None):
    if yaml_obj is None:
        parser = argparse.ArgumentParser(description="分類第一輪（YAML 版）")
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="YAML config 路徑"
        )
        args = parser.parse_args()

        cfg = load_yaml_config(args.config)
    else:
        cfg = yaml_obj

    task_cfg = cfg["task"]
    path_cfg = cfg["paths"]
    model_cfg = cfg["model"]
    norm_cfg = cfg["normalize"]
    vote_cfg = cfg["vote"]
    blank_cfg = cfg["blank_rule"]
    name_cfg = cfg["names"]
    regex_cfg = cfg["regex"]
    csv_cfg = cfg["csv"]

    # -------------------------
    # A. 任務識別 / 版本資訊
    # -------------------------
    MAP_NUMBER = str(task_cfg["map_number"])
    map_dir = str(task_cfg["map_dir"])
    ROUND = int(task_cfg["round"])

    DATA_NAME = str(task_cfg["data_name"])
    METHOD = str(task_cfg["method"])
    TRAIN_MODEL_METHOD = str(task_cfg["train_model_method"])
    num_workers = int(task_cfg.get("num_workers"))
    underscore_replace = str(task_cfg.get("underscore_replace"))

    # -------------------------
    # B. 輸入 / 輸出路徑
    # -------------------------
    IN_ROOTS = list(path_cfg["in_roots"])
    OUTPUT_ROOT = str(path_cfg["output_root"])
    CKPT_PATH = str(path_cfg["ckpt_path"])
    TIF_OUTPUT_DIR = Path(path_cfg.get("tif_output_dir"))

    # -------------------------
    # C. 模型 / 推論參數
    # -------------------------
    MODEL_NAME = str(model_cfg["model_name"])

    PATCH_H = int(model_cfg["patch_h"])
    PATCH_W = int(model_cfg["patch_w"])
    STRIDE_H = int(model_cfg["stride_h"])
    STRIDE_W = int(model_cfg["stride_w"])

    BATCH_SIZE = int(model_cfg["batch_size"])
    CONF_TH = float(model_cfg["conf_th"])

    gpu_ids = list(set(cfg["model"]["gpu_ids"].split(',')))
    num_gpus = len(gpu_ids)
    cls_procs = int(cfg["model"]["cls_procs"])

    USE_CUDA_IF_AVAILABLE = bool(model_cfg["use_cuda_if_available"])

    # -------------------------
    # D. 正規化參數
    # -------------------------
    MEAN = list(norm_cfg["mean"])
    STD = list(norm_cfg["std"])

    # -------------------------
    # E. 投票 / 決策參數
    # -------------------------
    MAJORITY_TH = float(vote_cfg["majority_th"])
    MIN_PATCH = int(vote_cfg["min_patch"])

    MIN_NONBLANK = int(vote_cfg["min_nonblank"])
    STRONG_MIN = int(vote_cfg["strong_min"])

    # -------------------------
    # F. 白底 / blank / 面積規則
    # -------------------------
    WHITE_BGR = tuple(blank_cfg["white_bgr"])
    BLANK_NAME = str(blank_cfg["blank_name"])

    SMALL_THRESHOLD_EXTRA_W = int(blank_cfg["small_threshold_extra_w"])

    HALF_AREA = (PATCH_H * PATCH_W) // 2
    SMALL_THRESHOLD = PATCH_H * PATCH_W + PATCH_H * SMALL_THRESHOLD_EXTRA_W

    # -------------------------
    # G. 資料夾名稱 / 檔名規則
    # -------------------------
    DIRNAME_MID = str(name_cfg["dirname_mid"])

    DIRNAME_CORRECT = str(name_cfg["dirname_correct"])
    DIRNAME_MERGED = str(name_cfg["dirname_merged"])
    DIRNAME_ERROR = str(name_cfg["dirname_error"])
    DIRNAME_ERROR_POSTMERGE_TMP = str(name_cfg["dirname_error_postmerge_tmp"])

    ERROR_CSV_NAME = str(name_cfg["error_csv_name"])
    SKIPPED_CSV_NAME = str(name_cfg["skipped_csv_name"])

    MERGED_CANVAS_FILENAME_TEMPLATE = str(name_cfg["merged_canvas_filename_template"])
    ERROR_IMAGE_FILENAME_TEMPLATE = str(name_cfg["error_image_filename_template"])
    COMPOSITE_SMALL_FILENAME_TEMPLATE = str(name_cfg["composite_small_filename_template"])
    SOURCE_DESC_TEMPLATE = str(name_cfg["source_desc_template"])

    # -------------------------
    # H. tile / regex 規則
    # -------------------------
    TILE_GLOB_PATTERN = str(regex_cfg["tile_glob_pattern"])
    TILE_DIR_REGEX = str(regex_cfg["tile_dir_regex"])
    TILE_COORD_REGEX = str(regex_cfg["tile_coord_regex"])

    # -------------------------
    # I. CSV 欄位
    # -------------------------
    ERROR_CSV_HEADER = list(csv_cfg["error_csv_header"])
    SKIPPED_CSV_HEADER = list(csv_cfg["skipped_csv_header"])

    # =========================================================
    # 1) 工具函式
    # =========================================================

    def is_white_mask(img_np: np.ndarray) -> np.ndarray:
        """嚴格白判定：三通道皆等於 WHITE_BGR 才算白。"""
        b, g, r = WHITE_BGR
        return (
            (img_np[..., 0] == b) &
            (img_np[..., 1] == g) &
            (img_np[..., 2] == r)
        )


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


    def iter_tiles(map_root: Path, imgs_nonwhite):
        """yield (ty, tx, [mid_paths])，並於此先依非白像素數由大到小排序"""
        tile_dir_re = re.compile(TILE_DIR_REGEX)

        for tile_dir in sorted(map_root.glob(TILE_GLOB_PATTERN)):
            m = tile_dir_re.match(tile_dir.name)
            if not m:
                continue

            ty, tx = int(m.group(1)), int(m.group(2))
            mid_dir = tile_dir / DIRNAME_MID
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
            sorted_mids = [p for _, p in areas]
            yield ty, tx, sorted_mids


    def build_canvas_size(tiles_list) -> Tuple[int, int, int, int]:
        """掃描第一張 mid 估算畫布大小與單張 tile H,W"""
        Hmax = Wmax = 0
        tile_h = tile_w = None

        for ty, tx, mids in tiles_list:
            if not mids:
                continue
            img = cv2.imread(str(mids[0]))
            if img is None:
                continue

            h, w = img.shape[:2]
            if tile_h is None:
                tile_h = h
            if tile_w is None:
                tile_w = w

            Hmax = max(Hmax, ty + h)
            Wmax = max(Wmax, tx + w)

        return Hmax, Wmax, (tile_h or 0), (tile_w or 0)


    def batched_infer(
        patches: List[Image.Image],
        model,
        val_tf,
        idx_to_label,
        device
    ) -> Tuple[List[str], List[float]]:
        labels, confs = [], []
        if not patches:
            return labels, confs

        tensors = [val_tf(p) for p in patches]
        batch = torch.stack(tensors, dim=0).to(device)

        logits_all = []
        with torch.no_grad():
            for i in range(0, len(batch), BATCH_SIZE):
                b = batch[i:i + BATCH_SIZE]
                logits_all.append(model(b))

        logits = torch.cat(logits_all, dim=0)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)

        conf = conf.detach().cpu().numpy().tolist()
        pred = pred.detach().cpu().numpy().tolist()

        for c, idx in zip(conf, pred):
            labels.append(idx_to_label[idx])
            confs.append(float(c))

        return labels, confs


    def classify_image_for_core(
        pil_img: Image.Image,
        model,
        val_tf,
        idx_to_label,
        device,
        conf_th: float = CONF_TH
    ) -> Dict:
        """
        新版決策（兩條互斥路：D vs B/C），且 patch 推論只做一次。

        前置：對每個非全白 patch 做推論，得到 pred/conf。
        有效票定義：非全白且 canon_label(pred) != blank 才算票。

        統計：
        total_nonblank  = 有效票數
        strong_nonblank = 有效票中 conf >= CONF_TH 的票數

        決策（互斥）：
        if total_nonblank >= MIN_PATCH:
            -> Branch D（主投票）
        elif total_nonblank >= MIN_NONBLANK and strong_nonblank >= STRONG_MIN:
            -> Branch B/C（強弱一致性）
        else:
            -> error
        """
        np_img = np.array(pil_img)
        H, W = np_img.shape[:2]

        if is_white_mask(np_img).mean() == 1.0:
            return {
                "status": "error",
                "assigned_core": BLANK_NAME,
                "strong_cnt": 0,
                "weak_cnt": 0,
                "core_counts": {},
                "reason": "retry_blank_image"
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
                "assigned_core": BLANK_NAME,
                "strong_cnt": 0,
                "weak_cnt": 0,
                "core_counts": {},
                "reason": "no_nonwhite_patches"
            }

        labels, confs = batched_infer(
            patches=patches,
            model=model,
            val_tf=val_tf,
            idx_to_label=idx_to_label,
            device=device
        )

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
                "assigned_core": BLANK_NAME,
                "strong_cnt": 0,
                "weak_cnt": 0,
                "core_counts": {},
                "reason": f"all_pred_blank(nonwhite_patches={len(patches)})"
            }

        # Branch D：主投票
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
                    "reason": f"branch_D_majority_{ratio:.3f}(hit={hit}/{total_nonblank})"
                }
            else:
                return {
                    "status": "error",
                    "assigned_core": cand,
                    "strong_cnt": strong_nonblank,
                    "weak_cnt": weak_nonblank,
                    "core_counts": dict(core_counts),
                    "reason": f"branch_D_fail_{ratio:.3f}(hit={hit}/{total_nonblank})"
                }

        # Branch B/C：強弱一致性
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
                        "reason": "branch_BC_ok(strong_intersection)"
                    }
                else:
                    return {
                        "status": "error",
                        "assigned_core": cand,
                        "strong_cnt": strong_nonblank,
                        "weak_cnt": weak_nonblank,
                        "core_counts": dict(core_counts),
                        "reason": f"branch_BC_fail_weak_mismatch({weak_mismatch})"
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
                        "reason": "branch_BC_ok(all_intersection)"
                    }
                else:
                    return {
                        "status": "error",
                        "assigned_core": cand,
                        "strong_cnt": strong_nonblank,
                        "weak_cnt": weak_nonblank,
                        "core_counts": dict(core_counts),
                        "reason": f"branch_BC_fail_weak_mismatch_all({weak_mismatch})"
                    }

            return {
                "status": "error",
                "assigned_core": None,
                "strong_cnt": strong_nonblank,
                "weak_cnt": weak_nonblank,
                "core_counts": dict(core_counts),
                "reason": "branch_BC_fail_no_intersection"
            }

        return {
            "status": "error",
            "assigned_core": None,
            "strong_cnt": strong_nonblank,
            "weak_cnt": weak_nonblank,
            "core_counts": dict(core_counts),
            "reason": f"insufficient_votes(total={total_nonblank}, strong={strong_nonblank}, blank_pred={blank_pred_cnt})"
        }


    def overlay_new_pixels(
        dst: np.ndarray,
        src: np.ndarray,
        y: int,
        x: int,
        confirmed_any: np.ndarray
    ):
        """
        僅把 src 的「非白 且 未被確認」像素貼到 dst，
        並回傳這次新覆蓋的布林遮罩（在 dst 的座標系）。
        """
        h, w = src.shape[:2]
        patch_dst = dst[y:y + h, x:x + w]
        patch_conf = confirmed_any[y:y + h, x:x + w]
        nonwhite = ~is_white_mask(src)
        newmask = nonwhite & (~patch_conf)
        patch_dst[newmask] = src[newmask]
        return newmask


    def append_reason(row, header_idx_map, header_len, extra):
        if len(row) < header_len:
            row += [""] * (header_len - len(row))
        row[header_idx_map["reason"]] = (row[header_idx_map["reason"]] or "") + extra


    # =========================================================
    # 2) 載入模型與轉換
    # =========================================================
    workers = [cls_worker.remote(gpu_ids[i%num_gpus] ,CKPT_PATH, MODEL_NAME, PATCH_H, PATCH_W, MEAN, STD, BATCH_SIZE, cfg, i) 
               for i in range(num_gpus*cls_procs)]
    models = ActorPool(workers)
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

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
        img_size=(PATCH_H, PATCH_W),
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    device = torch.device(
        "cuda" if (USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()) else "cpu"
    )
    model.to(device)

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


    # =========================================================
    # 3) 主流程（多地圖）
    # =========================================================

    OUT = Path(OUTPUT_ROOT)
    tile_coord_re = re.compile(TILE_COORD_REGEX)

    for map_root_str in IN_ROOTS:
        map_root = Path(map_root_str)

        # 例：/…/sam1/切割法/{train_model_method}/只有-17/round_1
        # 輸出到：OUTPUT_ROOT/{train_model_method}/只有-17/round_{ROUND}
        method_dir = map_root.parent.parent.name
        group_dir = map_root.parent.name

        out_map_dir = OUT / method_dir / group_dir / f"round_{ROUND}"

        correct_dir = out_map_dir / DIRNAME_CORRECT
        merged_dir = correct_dir / DIRNAME_MERGED
        error_dir = out_map_dir / DIRNAME_ERROR

        TIF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        merged_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)
        with open(map_root/'imgs_nonwhite.pkl', 'rb') as f:
            imgs_nonwhite = pickle.load(f)   
        tiles_list = list(iter_tiles(map_root, imgs_nonwhite))
        Hc, Wc, tile_h, tile_w = build_canvas_size(tiles_list)
        if Hc == 0 or Wc == 0:
            print(f"[WARN] 找不到 mid")
            continue

        canvases: Dict[str, np.ndarray] = defaultdict(
            lambda: np.full((Hc, Wc, 3), 255, np.uint8)
        )
        confirmed_any = np.zeros((Hc, Wc), dtype=bool)
        confirmed_id = np.zeros((Hc, Wc), dtype=np.int32)

        err_rows = [ERROR_CSV_HEADER.copy()]
        skip_rows = [SKIPPED_CSV_HEADER.copy()]

        total_mids = sum(len(mids) for _, _, mids in tiles_list)
        for ty, tx, mid_list in tiles_list:
            tile_y = slice(ty, ty + tile_h)
            tile_x = slice(tx, tx + tile_w)
            models.submit(lambda a, v: a.infer.remote(*v), (mid_list, ty, tx, Hc, Wc, map_root, tile_h, tile_w))

        pbar = tqdm(total=len(tiles_list), desc=f"{MAP_NUMBER} round{ROUND} tiles")
        for _ in tiles_list:
            ty, tx, confirmed_id_tile, confirmed_any_tile, canvases_tile, err_rows_tile, skip_rows_tile = models.get_next_unordered()
            tile_y = slice(ty, ty + tile_h)
            tile_x = slice(tx, tx + tile_w)
            confirmed_id[tile_y, tile_x] = confirmed_id_tile
            confirmed_any[tile_y, tile_x] = confirmed_any_tile
            err_rows.extend(err_rows_tile)
            skip_rows.extend(skip_rows_tile)
            for k,v in canvases_tile.items():
                canvases[k][tile_y, tile_x] = v[tile_y, tile_x]
            pbar.update()
        pbar.close()

        error_csv = error_dir / ERROR_CSV_NAME
        with open(error_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(err_rows)

        skipped_csv = error_dir / SKIPPED_CSV_NAME
        with open(skipped_csv, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows(skip_rows)

        merged_dir.mkdir(parents=True, exist_ok=True)
        for core, canvas in canvases.items():
            with open(f"{map_dir}/{MAP_NUMBER}.offset", 'r') as f:
                h, w, offset_x, offset_y = map(int, f.read().split())
            poly_pred = ~is_white_mask(canvas)
            mask=np.zeros((h, w), dtype=bool)
            mask[offset_y:offset_y+poly_pred.shape[0], offset_x:offset_x+poly_pred.shape[1]] = poly_pred
            with rasterio.open(TIF_OUTPUT_DIR / f"{MAP_NUMBER}_{core.replace(underscore_replace, '_')}_poly.tif", 'w', driver='GTiff', compress='lzw', height=h, width=w,count=1, dtype=np.uint8) as fh:
                fh.write(mask[None,:,:].astype(np.uint8))
            out_path = merged_dir / MERGED_CANVAS_FILENAME_TEMPLATE.format(core=core.replace(underscore_replace, '_'))
            cv2.imwrite(str(out_path), canvas)

        # =====================================================
        # 4) 小圖整合（postmerge）
        # =====================================================

        src_error_dir = error_dir
        src_error_csv = src_error_dir / ERROR_CSV_NAME
        src_skipped_csv = src_error_dir / SKIPPED_CSV_NAME

        dst_error_dir = out_map_dir / DIRNAME_ERROR_POSTMERGE_TMP
        if dst_error_dir.exists():
            shutil.rmtree(dst_error_dir)
        dst_error_dir.mkdir(parents=True, exist_ok=True)

        dst_error_csv = dst_error_dir / ERROR_CSV_NAME
        dst_skipped_csv = dst_error_dir / SKIPPED_CSV_NAME

        header = ERROR_CSV_HEADER.copy()
        rows = []

        base_csv = dst_error_csv if dst_error_csv.exists() else src_error_csv
        if base_csv.exists():
            with open(base_csv, "r", newline="", encoding="utf-8-sig") as f:
                reader = list(csv.reader(f))
            if reader:
                if reader[0] and reader[0][0] == "map":
                    header = reader[0]
                    rows = reader[1:]
                else:
                    rows = reader

        idx = {k: i for i, k in enumerate(header)}

        if src_skipped_csv.exists():
            shutil.copy2(src_skipped_csv, dst_skipped_csv)

        # 0) 先把既有 composite_small 複製到 tmp，避免被漏掉
        for p in src_error_dir.glob("*_composite_small.png"):
            shutil.copy2(p, dst_error_dir / p.name)

            m = tile_coord_re.search(p.name)
            if not m:
                continue

            ty, tx = int(m.group(1)), int(m.group(2))
            comp_name = p.name

            if not any(len(r) >= len(header) and r[idx["mid_file"]] == comp_name for r in rows):
                new_row = [""] * len(header)
                if "map" in idx:
                    new_row[idx["map"]] = MAP_NUMBER
                if "tile_y" in idx:
                    new_row[idx["tile_y"]] = str(ty)
                if "tile_x" in idx:
                    new_row[idx["tile_x"]] = str(tx)
                if "mid_file" in idx:
                    new_row[idx["mid_file"]] = comp_name
                if "status" in idx:
                    new_row[idx["status"]] = "retry"
                if "assigned_core" in idx:
                    new_row[idx["assigned_core"]] = ""
                if "strong_cnt" in idx:
                    new_row[idx["strong_cnt"]] = "0"
                if "weak_cnt" in idx:
                    new_row[idx["weak_cnt"]] = "0"
                if "top_core_counts" in idx:
                    new_row[idx["top_core_counts"]] = ""
                if "reason" in idx:
                    new_row[idx["reason"]] = "postmerge_copied_existing"
                rows.append(new_row)

        pngs = sorted(src_error_dir.glob("*.png"))

        small_by_tile = defaultdict(list)
        large_to_copy = []

        base_comp_by_tile = {}
        for p in dst_error_dir.glob("*_composite_small.png"):
            m = tile_coord_re.search(p.name)
            if m:
                base_comp_by_tile[(int(m.group(1)), int(m.group(2)))] = p

        for p in src_error_dir.glob("*_composite_small.png"):
            m = tile_coord_re.search(p.name)
            if m and (int(m.group(1)), int(m.group(2))) not in base_comp_by_tile:
                base_comp_by_tile[(int(m.group(1)), int(m.group(2)))] = p

        for p in tqdm(pngs, desc="Scan & split small/large"):
            if p.name.endswith("_composite_small.png"):
                continue

            m = tile_coord_re.search(p.name)
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
            shutil.copy2(p, dst_error_dir / p.name)

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

            comp_name = COMPOSITE_SMALL_FILENAME_TEMPLATE.format(
                map_name=MAP_NUMBER,
                ty=ty,
                tx=tx
            )
            comp_path = dst_error_dir / comp_name
            cv2.imwrite(str(comp_path), base_img)

            row_found = None
            for r in rows:
                if len(r) >= len(header) and r[idx["mid_file"]] == comp_name:
                    row_found = r
                    break

            if row_found:
                append_reason(row_found, idx, len(header), "|postmerge_added")
                updated_tiles += 1
            else:
                new_row = [""] * len(header)
                if "map" in idx:
                    new_row[idx["map"]] = MAP_NUMBER
                if "tile_y" in idx:
                    new_row[idx["tile_y"]] = str(ty)
                if "tile_x" in idx:
                    new_row[idx["tile_x"]] = str(tx)
                if "mid_file" in idx:
                    new_row[idx["mid_file"]] = comp_name
                if "status" in idx:
                    new_row[idx["status"]] = "retry"
                if "assigned_core" in idx:
                    new_row[idx["assigned_core"]] = ""
                if "strong_cnt" in idx:
                    new_row[idx["strong_cnt"]] = "0"
                if "weak_cnt" in idx:
                    new_row[idx["weak_cnt"]] = "0"
                if "top_core_counts" in idx:
                    new_row[idx["top_core_counts"]] = ""

                sources = ";".join([p.name for p in file_list])
                reason = f"postmerge_small|sources={sources}"
                if "reason" in idx:
                    new_row[idx["reason"]] = reason

                rows.append(new_row)
                added_tiles += 1

            for p in file_list:
                for r in rows:
                    if len(r) >= len(header) and r[idx["mid_file"]] == p.name:
                        append_reason(
                            r, idx, len(header),
                            f"|postmerge_collapsed_into={comp_name}"
                        )
                        break

        with open(dst_error_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"[postmerge] 完成（tmp）：{dst_error_dir}")
        print(f"  - 大圖（>= 門檻）已複製：{len(large_to_copy)} 張")
        print(f"  - 合併小圖：貼入次數 {merged_count}；新增 tile：{added_tiles}；更新 tile：{updated_tiles}")
        print(f"  - tmp errors.csv / skipped.csv 已就緒：{dst_error_csv}")

        # =====================================================
        # 5) 覆蓋原本的錯誤資料夾
        # =====================================================
        if error_dir.exists():
            shutil.rmtree(error_dir)
        shutil.move(str(dst_error_dir), str(error_dir))
        print(f"[postmerge] ✅ 已覆蓋：{error_dir}（主體舊內容已刪除，換成 postmerge 結果）")

    print("\n✅ 全部完成")