
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config

current_map  = config.args.map
method = config.args.cls_method
ckpt_name = config.args.cls_ckpt # "fixed_train_data_20260316_171401"
data_name = config.args.cls_data # "fixed_train_data_20260410_091015"
target_group = config.args.cls_eval_target.replace('legends', 'FG').replace('others', 'BGC')  #當前檢測項目，BGC、FG




# -*- coding: utf-8 -*-

#══════════════ 0. 必要匯入 ═════════════════════════════
import os, cv2, torch, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import Counter
from typing import Tuple, List, Dict
import torch.nn as nn
from torchvision import transforms
import timm
import pickle

from sklearn.metrics import f1_score, accuracy_score
import json, re


#══════════════ 1. 參數（全部集中在這裡） ═════════════════════════════

# ---- I/O paths ----
JSON_PATH   = f"{config.args.cls_dir}/stage_3/{method}/{current_map}/{data_name}/train_data.json"
CKPT_PATH = f"{config.args.model_dir}/{method}/{current_map}/{ckpt_name}/best_model.pth"
IN_DIR      = f"{config.args.cls_key_dir}/{current_map}_whitebg_seg"
OUT_DIR     = f"reports/{method}/{target_group}/{current_map}/{data_name}/labeled_maps_256x256"
CSV_OUT     = f"reports/{method}/{target_group}/{current_map}/{data_name}/cls_report_image_view_white_le50.csv"

# ---- image / patch ----
PATCH_H, PATCH_W = 56, 98
TARGET_SIZE      = (PATCH_H, PATCH_W)
STRIDE_H, STRIDE_W = PATCH_H, PATCH_W
HALF_AREA        = (PATCH_H * PATCH_W) // 2   # ★ 不足此面積的 patch 不納入統計

# ---- blank / confidence threshold ----
WHITE_TH_RATIO = 0.50      # >50% = 空白 patch
WHITE_TH_PIXEL = 245       # RGB 全通道 > 245 視為白色
CONF_TH        = 0.30      # 信心低於此 → unclassified
BLANK_LABELS   = {"blank"} # blank 類別集合

# ---- model / preprocessing ----
TIMM_MODEL_NAME = "eva02_small_patch14_336.mim_in22k_ft_in1k"
TIMM_PRETRAINED = False

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ---- file extensions ----
VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


#══════════════ 2. 映射與正規化工具（使用前必先宣告） ═════════════════════════════
# 複合標籤映射容器
_COMPONENTS_SET_TO_CANON = {}   # frozenset(parts) -> 'A-B(-C)'
_PART_TO_BEST_CANON = {}        # 'Mk' -> 'Mag-Mk'（若重疊，取子項數最多者）

def _clean_base_token(tok: str) -> str:
    """子 token 僅保留開頭英文字母；支援 Mk_001、Mag002、Qo-1 等寫法。"""
    if tok is None: return None
    s = str(tok).strip()
    s = s.split('_', 1)[0]
    m = re.match(r'^([A-Za-z]+)', s)
    return m.group(1) if m else s

def normalize_core(name: str) -> str:
    """
    對單項或複合（A-B(-C)）做正規化：
    - 單一子項：若可映射到複合 → 提升成複合
    - 多子項：依 parts 集合查 _COMPONENTS_SET_TO_CANON（順序無關、支援反序）
    - 都找不到就回清洗後原樣（用 '-' 接回）
    """
    if name is None:
        return None
    s = str(name).strip()
    s = Path(s).name
    s = s.split('.', 1)[0]
    parts = [p for p in (_clean_base_token(t) for t in s.split('-')) if p]
    if not parts:
        return None
    if len(parts) == 1:
        p = parts[0]
        return _PART_TO_BEST_CANON.get(p, p)
    canon = _COMPONENTS_SET_TO_CANON.get(frozenset(parts))
    return canon if canon else '-'.join(parts)

def core_of(label: str) -> str:
    """把完整標籤（可能含 '_idx'、含 '-'）正規化成 canonical core。"""
    if not label: return None
    base = str(label).split('_', 1)[0]  # 先砍掉 '_尾綴'
    return normalize_core(base)

def same_group(pred_label: str, gt_label: str) -> bool:
    """pred 與 gt 先做 canonical；相等即視為同群組（含子類→複合提升）。"""
    return core_of(pred_label) == core_of(gt_label)

def _try_load_label_map(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    lm = pickle.load(f)
                if isinstance(lm, dict) and len(lm) > 0:
                    print(f"[info] 讀到 label_to_idx：{p}（{len(lm)} 類）")
                    return lm
            except Exception as e:
                print(f"[warn] 讀取 {p} 失敗：{e}")
    return None


#══════════════ 3. 讀取 ckpt / label mapping / 建立 canonical 映射 ═════════════════════════════
ckpt = torch.load(CKPT_PATH, map_location="cpu")
label_to_idx = ckpt["label_to_idx"]
idx_to_label = {v: k for k, v in label_to_idx.items()}
NUM_CLASSES = len(label_to_idx)

def _extract_labels_from_json_file(path: str) -> list:
    """
    從 train_data.json 抓原始類別：
    - 你的格式: {"classes": { "Kl_17": {...}, "Mag-Mk_17_13": {...}, ... }}
    - 先拿 key 當 raw name
    - 再做一次基本清洗（去掉 _ 後綴與數字，保留 '-' 結構）
      e.g. "Mag-Mk_17_13" -> "Mag-Mk"
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            jd = json.load(f)

        classes_raw = []

        if isinstance(jd, dict):
            # 1) 你的主要格式：classes 是 dict
            if "classes" in jd and isinstance(jd["classes"], dict):
                classes_raw = list(jd["classes"].keys())
            # 2) 如果未來改成 list 也能支援
            elif "classes" in jd and isinstance(jd["classes"], (list, tuple)):
                classes_raw = list(jd["classes"])
            else:
                # 備用：其他常見鍵名
                for k in ["class_names", "labels", "label_names"]:
                    if k in jd and isinstance(jd[k], (list, tuple)):
                        classes_raw = list(jd[k])
                        break
        elif isinstance(jd, list):
            classes_raw = list(jd)

        if not classes_raw:
            return []

        out, seen = [], set()
        for c in classes_raw:
            s = str(c).strip()
            if not s or s.lower() == "nan":
                continue
            parts = [p for p in (_clean_base_token(t) for t in s.split('-')) if p]
            if not parts:
                continue
            cc = '-'.join(parts)
            if cc not in seen:
                seen.add(cc)
                out.append(cc)

        return out

    except Exception as e:
        print(f"[warn] 讀 JSON 失敗：{e}")
        return []

# 先從 JSON 取原始類別；沒有就用 ckpt 內的 label 名稱
_raw_labels = _extract_labels_from_json_file(JSON_PATH)
if not _raw_labels:
    seen = set(); _raw_labels = []
    for lbl in idx_to_label.values():
        parts = [p for p in (_clean_base_token(t) for t in str(lbl).split('-')) if p]
        if parts:
            cc = '-'.join(parts)
            if cc not in seen:
                seen.add(cc); _raw_labels.append(cc)

# 建立複合映射
_COMPONENTS_SET_TO_CANON.clear()
_PART_TO_BEST_CANON.clear()
_part_to_cands = {}
for lbl in _raw_labels:
    parts = lbl.split('-')
    if len(parts) >= 2:
        _COMPONENTS_SET_TO_CANON.setdefault(frozenset(parts), lbl)
        for p in parts:
            _part_to_cands.setdefault(p, []).append(lbl)

def _comp_len(x: str) -> int:
    return len(x.split('-'))

for p, cands in _part_to_cands.items():
    cands_sorted = sorted(cands, key=lambda x: (-_comp_len(x), _raw_labels.index(x)))
    _PART_TO_BEST_CANON[p] = cands_sorted[0]


# 取出 state_dict（嚴格用同一個 ckpt 的權重）
state_dict = ckpt["state_dict"]
# 若當時是 DataParallel 存的，去掉 'module.' 前綴
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

# （可選，但很有用）頭部維度檢查，避免類別數不對還悄悄跑下去
if "head.weight" in state_dict:
    expect_out = NUM_CLASSES
    assert state_dict["head.weight"].shape[0] == expect_out, \
        f"head.weight 的輸出數 {state_dict['head.weight'].shape[0]} 與期待 {expect_out} 不符（NUM_CLASSES={NUM_CLASSES}）"


#══════════════ 4. 還原模型 ═════════════════════════════
model = timm.create_model(
    TIMM_MODEL_NAME,
    pretrained=TIMM_PRETRAINED,
    num_classes=NUM_CLASSES,
    img_size=TARGET_SIZE,   # ← 關鍵：改這個
)
model.load_state_dict(state_dict, strict=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

#══════════════ 6. 驗證單張影像 ═════════════════════════════
def validate_one_image(img_path: str,
                         gt_core: str,
                         conf_th: float = 0.0) -> Tuple[Dict, List[str], List[str]]:
    """
    回傳 dict 統計（單張圖片）：
      • img             圖檔檔名
      • total           全部 patch 數（含邊角）
      • checked         實際檢測 patch 數（面積 ≥ 一半）
      • blank_area      空白 patch（白色占比 > 50%）
      • content_area    有內容 patch（白色占比 ≤ 50%）
      • blank(pred)     空白 → 預測 blank
      • blank_miss      空白 → 預測非 blank
      • correct         彩色 → 分類正確
      • wrong           彩色 → 分類錯誤
      • unclassified    彩色 → 低信心 (conf < CONF_TH)
      • quality_all     1 - (wrong + blank_miss + unclassified) / checked
      • quality_content 1 - (wrong + unclassified) / checked
      • cls_acc         correct / content_area
      • cls_f1          自定義 F1（tolerant）
      • wrong_detail    各錯分類別統計字串
      • unclassified_detail  低信心統計字串
      • blank_miss_detail    空白漏判統計字串
    """
    pil_img = Image.open(img_path).convert("RGB")
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np_img  = np.array(pil_img)
    H, W, _ = np_img.shape

    gt_core = core_of(Path(img_path).stem)

    total = checked = 0
    blank_area = content_area = 0
    blank_hit = blank_miss = 0
    content_hit = content_miss = unclassified = 0
    wrong_pred_counts = Counter()
    unclassified_pred_counts = Counter()
    blank_miss_pred_counts = Counter()

    y_true, y_pred_tol = [], []
    patchs=[]
    for y in range(0, H, STRIDE_H):
        for x in range(0, W, STRIDE_W):

            # 1) edge patch 面積判定
            w_eff, h_eff = min(PATCH_W, W - x), min(PATCH_H, H - y)
            if w_eff * h_eff < HALF_AREA:
                continue

            # 2) 取 / 補 patch
            patch = pil_img.crop((x, y, x + w_eff, y + h_eff))
            if patch.size != (PATCH_W, PATCH_H):
                pad = Image.new("RGB", (PATCH_W, PATCH_H), (255, 255, 255))
                pad.paste(patch, (0, 0))
                patch = pad
            patchs.append(patch)
    batch_size=config.args.cls_bs
    confs = np.zeros(len(patchs), dtype=np.float32)
    preds = np.zeros(len(patchs), dtype=np.int32)
    for i in range(0, len(patchs), batch_size):
        batch = patchs[i : i + batch_size]
        batch = torch.concat([val_tf(p).unsqueeze(0).to(device) for p in batch])
        with torch.no_grad():
            logits = model(batch)
            prob   = torch.softmax(logits, dim=1)
            conf, pred = prob.max(1)
            conf = conf.cpu().numpy()
            pred = pred.cpu().numpy()
            confs[i : i + batch_size] = conf
            preds[i : i + batch_size] = pred

    for y in range(0, H, STRIDE_H):
        for x in range(0, W, STRIDE_W):
            total += 1

            # 1) edge patch 面積判定
            w_eff, h_eff = min(PATCH_W, W - x), min(PATCH_H, H - y)
            if w_eff * h_eff < HALF_AREA:
                continue
            checked += 1

            # 2) 取 / 補 patch
            patch = pil_img.crop((x, y, x + w_eff, y + h_eff))
            if patch.size != (PATCH_W, PATCH_H):
                pad = Image.new("RGB", (PATCH_W, PATCH_H), (255, 255, 255))
                pad.paste(patch, (0, 0))
                patch = pad

            # 3) 白色比例
            arr         = np.array(patch)
            white_ratio = (arr > WHITE_TH_PIXEL).all(axis=2).mean()
            is_blank    = white_ratio > WHITE_TH_RATIO

            # 4) 推論
            conf = confs[total-1]
            pred = preds[total-1]

            label = idx_to_label[pred.item()]
            pred_core_full = label
            pred_core = core_of(label)

            # --- 4. 顏色邏輯 ---
            if label in BLANK_LABELS:
                color = (255,   0,   0)        # BGR → 藍
            elif conf < conf_th:
                color = (  0, 255, 255)        # 黃
            elif same_group(pred_core_full, gt_core):
                color = (  0, 255,   0)        # 綠
            else:
                color = (  0,   0, 255)        # 紅

            cv2.putText(cv_img,
                        f"{label} {conf*100:.1f}%",
                        (x + 2, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)

            # 6) 分類與歸納
            if is_blank:
                blank_area += 1
                if label in BLANK_LABELS and conf >= CONF_TH:
                    blank_hit  += 1
                else:
                    blank_miss += 1
                    if label in BLANK_LABELS:
                        key = "blank_lowconf"
                    else:
                        key = pred_core
                    blank_miss_pred_counts[key] += 1
                continue

            content_area += 1
            y_true.append(gt_core)

            if conf < CONF_TH:
                y_pred_tol.append(pred_core)
                unclassified += 1
                unclassified_pred_counts[pred_core] += 1
            else:
                if same_group(pred_core_full, gt_core):
                    content_hit += 1
                    y_pred_tol.append(gt_core)
                else:
                    content_miss += 1
                    wrong_pred_counts[pred_core] += 1
                    y_pred_tol.append(pred_core)

    wrong_detail = "; ".join(f"{k}:{v}" for k, v in wrong_pred_counts.items())
    unclassified_detail = "; ".join(f"{k}:{v}" for k, v in unclassified_pred_counts.items())
    blank_miss_detail = "; ".join(f"{k}:{v}" for k, v in blank_miss_pred_counts.items())

    if checked == 0:
        quality_all = quality_content = np.nan
    else:
        quality_all = 1 - (content_miss + blank_miss + unclassified) / checked
        quality_content = 1 - (content_miss + unclassified) / checked

    if content_area:
        cls_acc = content_hit / content_area
        precision = (content_hit / (content_hit + content_miss + unclassified)
                     if (content_hit + content_miss + unclassified) else 0)
        recall = content_hit / (content_hit + content_miss + unclassified) if (content_hit + content_miss + unclassified) else 0
        cls_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    else:
        cls_acc = cls_f1 = 0.0

    return {
        "img"          : Path(img_path).name,
        "total"        : total,
        "checked"      : checked,
        "blank_area"   : blank_area,
        "content_area" : content_area,
        "blank(pred)"  : blank_hit,
        "blank_miss"   : blank_miss,
        "correct"      : content_hit,
        "wrong"        : content_miss,
        "unclassified" : unclassified,
        "quality_all"  : round(quality_all, 4),
        "quality_content"  : round(quality_content, 4),
        "cls_acc"      : round(cls_acc, 4),
        "cls_f1"       : round(cls_f1, 4),
        "wrong_detail" : wrong_detail,
        "unclassified_detail": unclassified_detail,
        "blank_miss_detail": blank_miss_detail,
    }, y_true, y_pred_tol, cv_img


#══════════════ 7. 批次驗證 ═════════════════════════════
def batch_validate(in_dir: str, csv_path: str = CSV_OUT) -> None:
    """
    逐張圖片跑 validate_one_image()，產生：
      • 每張圖的標註 PNG 到 OUT_DIR
      • 統計結果 DataFrame → console ＋ CSV
      • TOTAL 行包含 overall Accuracy 與 Weighted-F1
    """
    files = [os.path.join(in_dir, f) for f in sorted(os.listdir(in_dir))
             if f.lower().endswith(VALID_EXT)]
    if str(target_group) == "BGC":
        files = [item for item in files if item.endswith('BGC_seg.png')]
    else:
        files = [item for item in files if not item.endswith('BGC_seg.png')]
    if not files:
        print("⚠️  資料夾裡沒找到圖片！")
        return

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    rows = []
    y_true_all, y_pred_all_tol = [], []
    for fp in tqdm(files, desc="Validating"):
        stats, y_t, y_p_tol, labeled = validate_one_image(fp, gt_core=core_of(Path(fp).stem),conf_th=CONF_TH)
        rows.append(stats)

        y_true_all.extend(y_t)
        y_pred_all_tol.extend(y_p_tol)

        pil_img = Image.open(fp).convert("RGB")
        cv2.imwrite(os.path.join(OUT_DIR, f"{Path(fp).stem}_labeled.png"), labeled)

    df = pd.DataFrame(rows)

    overall_acc = accuracy_score(y_true_all, y_pred_all_tol)
    micro_f1    = f1_score(y_true_all, y_pred_all_tol, average="micro")
    if str(target_group) == "BGC":
        y_true_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in y_true_all]
        y_pred_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in y_pred_all_tol]
        macro_f1   = f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0)
    else:
        macro_f1   = f1_score(y_true_all, y_pred_all_tol, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_all, y_pred_all_tol, average="weighted")

    sum_cols = ["total", "checked", "blank_area", "content_area",
                "blank(pred)", "blank_miss", "correct",
                "wrong", "unclassified"]
    total_row = df[sum_cols].sum(numeric_only=True)

    checked_sum = total_row["checked"]
    if checked_sum:
        total_row["quality_all"] = (
            (df["quality_all"] * df["checked"]).fillna(0).sum() / checked_sum
        )
        total_row["quality_content"] = (
            (df["quality_content"] * df["checked"]).fillna(0).sum() / checked_sum
        )

    content_sum = total_row["content_area"]
    if content_sum:
        total_row["cls_acc"] = (
            (df["cls_acc"] * df["content_area"]).fillna(0).sum() / content_sum
        )
        total_row["cls_f1"] = (
            (df["cls_f1"] * df["content_area"]).fillna(0).sum() / content_sum
        )

    total_row["overall_acc"] = round(overall_acc, 4)
    total_row["micro_f1"]    = round(micro_f1, 4)
    total_row["macro_f1"]    = round(macro_f1, 4)
    total_row["weighted_f1"] = round(weighted_f1, 4)

    for col in ["overall_acc","micro_f1","macro_f1","weighted_f1"]:
        df[col] = np.nan

    total_row["wrong_detail"] = ""
    total_row["unclassified_detail"] = ""
    total_row["blank_miss_detail"] = ""

    total_row["img"] = "TOTAL"
    df.loc[len(df)] = total_row

    print("\n=== 結果總覽（每張圖；TOTAL 只有 tolerant 指標） ===")
    print(df.to_string(index=False))

    print("\n── Overall (tolerant only) ──")
    print(f"Overall Accuracy : {overall_acc:.4f}")
    print(f"Micro-F1         : {micro_f1:.4f}")
    print(f"Macro-F1         : {macro_f1:.4f}")
    print(f"Weighted-F1      : {weighted_f1:.4f}")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已儲存：{csv_path}")


##compute training setting metrics
batch_validate(IN_DIR, csv_path=CSV_OUT)


###compute depolying setting metrics
src_csv   = f"reports/{method}/{target_group}/{current_map}/{data_name}/cls_report_image_view_white_le50.csv"
out_csv   = f"reports/{method}/{target_group}/{current_map}/{data_name}/cls_report_vote_view.csv"

# --- blank / 特殊 key 設定 ---
BLANK_LABELS      = {"blank"}          # 如果未來有 none/white 再加進來
BLANK_LOWCONF_KEY = "blank_lowconf"
BLANK_GT_LABEL    = "__blank_gt__"     # 投票視角下的「真實空白」替代標籤

# --- TOTAL 行判定/顯示 ---
TOTAL_NAME = "TOTAL"                   # df["img"] == "TOTAL" 視為總結列（大小寫不拘）

# --- 票數統計輸出設定 ---
TOP_K = 5                              # TOTAL 的 vote_err_detail 只顯示前 K 名錯誤類別
ROUND_NDIGITS = 4                      # round(x, 4)

# --- parse_detail 行為參數 ---
DETAIL_SPLIT_REGEX = r"[;,|\n]+"
DETAIL_COLON_FULLWIDTH = "："
DETAIL_COLON = ":"
SPECIAL_PRESERVE_KEYS = {"bg", "bg_color","BGC"}  # parse_detail 中保留原樣的 key

# --- 欄位清單（print / export / sum）---
PRINT_COLS = [
    "img",
    "blank_miss_nonblank",
    "correct",
    "wrong_nonblank",
    "unclassified_nonblank",
    "vote_total",
    "vote_correct",
    "vote_err",
    "vote_err_detail",
    "vote_cls_acc",
    "vote_cls_f1",
]

sum_cols = [
    "blank_miss_nonblank",
    "correct",
    "wrong_nonblank",
    "unclassified_nonblank",
    "vote_total",
    "vote_correct",
    "vote_err",
]

cols_to_export = [
    "img",
    "blank_miss_nonblank",
    "correct",
    "wrong_nonblank",
    "unclassified_nonblank",
    "vote_total",
    "vote_correct",
    "vote_err",
    "vote_err_detail",
    "vote_cls_acc",
    "vote_cls_f1",
    "overall_acc",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
]

# =========================================================
# 1) 原本程式（功能不變：只把參數搬到上面）
# =========================================================

_COMPONENTS_SET_TO_CANON = {}
_PART_TO_BEST_CANON = {}

def _clean_base_token(tok: str) -> str:
    """子 token 僅保留開頭英文字母；支援 Mk_001、Mag002、Qo-1 等寫法。"""
    if tok is None: return None
    s = str(tok).strip()
    s = s.split('_', 1)[0]
    m = re.match(r'^([A-Za-z]+)', s)
    return m.group(1) if m else s

def normalize_core(name: str) -> str:
    """
    對單項或複合（A-B(-C)）做正規化：
    - 單一子項：若可映射到複合 → 提升成複合
    - 多子項：依 parts 集合查 _COMPONENTS_SET_TO_CANON（順序無關、支援反序）
    - 都找不到就回清洗後原樣（用 '-' 接回）
    """
    if name is None:
        return None
    s = str(name).strip()
    s = Path(s).name
    s = s.split('.', 1)[0]
    parts = [p for p in (_clean_base_token(t) for t in s.split('-')) if p]
    if not parts:
        return None
    if len(parts) == 1:
        p = parts[0]
        return _PART_TO_BEST_CANON.get(p, p)
    canon = _COMPONENTS_SET_TO_CANON.get(frozenset(parts))
    return canon if canon else '-'.join(parts)

def extract_class_list_from_json(jobj) -> list:
    """
    從 JSON 抓原始類別並做「僅清洗、不合併」：
    - 支援 {"classes": { "Kl_17": {...}, "Mag-Mk_17_13": {...}, ... }}
    - 先拿 key 當 raw name
    - 再用 _clean_base_token 清掉尾綴與數字，保留 '-' 結構
      e.g. "Mag-Mk_17_13" -> "Mag-Mk"
    """
    classes_raw = []

    if isinstance(jobj, dict):
        if "classes" in jobj and isinstance(jobj["classes"], dict):
            classes_raw = list(jobj["classes"].keys())
        elif "classes" in jobj and isinstance(jobj["classes"], (list, tuple)):
            classes_raw = list(jobj["classes"])
        else:
            for k in ["class_names", "labels", "label_names"]:
                if k in jobj and isinstance(jobj[k], (list, tuple)):
                    classes_raw = list(jobj[k])
                    break
    elif isinstance(jobj, list):
        classes_raw = list(jobj)

    if not classes_raw:
        return []

    seen, classes = set(), []
    for c in classes_raw:
        if c is None:
            continue
        s = str(c).strip()
        if not s or s.lower() == "nan":
            continue

        parts = [p for p in (_clean_base_token(t) for t in s.split('-')) if p]
        if not parts:
            continue
        cc = '-'.join(parts)
        if cc not in seen:
            seen.add(cc)
            classes.append(cc)

    return classes

# === 從 JSON 建立複合標籤映射：Mk -> Mag-Mk 這種 ===
with open(JSON_PATH, "r", encoding="utf-8") as f:
    jd = json.load(f)

labels = extract_class_list_from_json(jd)   # e.g. ["Kl", "Mag-Mk", "Mm", ...]

_COMPONENTS_SET_TO_CANON.clear()
_PART_TO_BEST_CANON.clear()
_part_to_cands = {}

for lbl in labels:
    parts = lbl.split('-')
    if len(parts) >= 2:
        key = frozenset(parts)
        _COMPONENTS_SET_TO_CANON.setdefault(key, lbl)
        for p in parts:
            _part_to_cands.setdefault(p, []).append(lbl)

def _comp_len(x: str) -> int:
    return len(x.split('-'))

for p, cands in _part_to_cands.items():
    # 優先子項數多的複合；平手時用 labels 裡較前面的
    cands_sorted = sorted(cands, key=lambda x: (-_comp_len(x), labels.index(x)))
    _PART_TO_BEST_CANON[p] = cands_sorted[0]


def core_of(label: str) -> str:
    """把完整標籤（可能含 '_idx'、含 '-'）正規化成 canonical core。"""
    if not label: return None
    base = str(label).split('_', 1)[0]
    return normalize_core(base)


def parse_detail(detail_str) -> Counter:
    """
    解析像 'Qy:12; Qo:3; blank_lowconf:2' 的字串成 Counter，
    並把一般類別名稱正規化成 canonical core（支援複合標籤）。
    特殊 key（blank_lowconf / blank 等）則保留原樣。
    """
    c = Counter()
    if detail_str is None or (isinstance(detail_str, float) and np.isnan(detail_str)):
        return c
    s = str(detail_str).strip()
    if not s:
        return c

    s = s.replace(DETAIL_COLON_FULLWIDTH, DETAIL_COLON)
    # 允許用 ; , | 換行 分隔
    for tok in re.split(DETAIL_SPLIT_REGEX, s):
        tok = tok.strip()
        if not tok:
            continue

        if ":" in tok:
            key_raw, num_str = tok.split(":", 1)
            key_raw = key_raw.strip()
            num_str = num_str.strip()
            try:
                n = int(round(float(num_str)))
            except ValueError:
                continue
        else:
            key_raw = tok
            n = 1

        if not key_raw:
            continue

        # 特殊 key 直接保留
        if key_raw == BLANK_LOWCONF_KEY:
            key = BLANK_LOWCONF_KEY
        elif key_raw in BLANK_LABELS or key_raw in SPECIAL_PRESERVE_KEYS:
            key = key_raw
        else:
            key = normalize_core(key_raw)

        if not key:
            continue

        c[key] += n

    return c


# ── 讀第一張表 ─────────────────────────────
df = pd.read_csv(src_csv)

# 把原始檔裡的 TOTAL 行先標記出來（如果沒有就全 False）
is_total = df["img"].astype(str).str.upper().eq(TOTAL_NAME)

# 後面所有「按每張圖計算」的東西，都只用非 TOTAL 的部分
df_body = df[~is_total].copy()

# 只保留真正有圖名的列
df_img = df_body[df_body["img"].notna()].copy()


# 新增欄位（投票視角細分）
df_img["blank_miss_nonblank"]   = 0

df_img["wrong_nonblank"]        = 0
df_img["unclassified_nonblank"] = 0   # 低信心 + 判錯 + 非 blank

# 四個「投票總結」欄位（統一用 vote_ 開頭）
df_img["vote_total"]        = 0        # 有投票的總票數
df_img["vote_correct"]      = 0        # 投對的票數（高信心 + 低信心）
df_img["vote_err"]          = 0        # 投錯的票數
df_img["vote_err_detail"]   = ""       # 投錯細節，例如 'Oku:3; Qy:2'

# 投票視角的分類指標
df_img["vote_cls_acc"]      = 0.0
df_img["vote_cls_f1"]       = 0.0

y_true_vote = []
y_pred_vote = []

# 🔴 全部圖片的錯誤類別總計（拿來算 TOTAL 的 vote_err_detail）
global_vote_err_counter = Counter()

for idx, row in df_img.iterrows():
    img_name = row["img"]
    gt_core = core_of(Path(img_name).stem)

    # 這張圖在「投票視角」的真實標籤：
    # - 一般情況：就是檔名的 core（例如 Kl_seg.png -> 'Kl'）
    # - 如果 core 是 blank / None，就當成 BLANK_GT_LABEL
    true_for_votes = gt_core if gt_core and gt_core not in BLANK_LABELS else BLANK_GT_LABEL

    # 高信心 correct（用原本欄位）
    correct_hi = int(row["correct"])

    # detail 欄位（已做 canonical）
    bmd = parse_detail(row.get("blank_miss_detail", ""))
    wd  = parse_detail(row.get("wrong_detail", ""))
    ud  = parse_detail(row.get("unclassified_detail", ""))

    # ===== 從各種 detail 重新算票數 =====
    # 空白 miss 中，扣掉 blank_lowconf & blank 類本身，只算「非 blank 類別的票」
    blank_miss_nonblank = sum(
        v for k, v in bmd.items()
        if k != BLANK_LOWCONF_KEY and k not in BLANK_LABELS
    )

    # 高信心錯：只算預測為非 blank 的
    wrong_nonblank = sum(
        v for k, v in wd.items()
        if k not in BLANK_LABELS
    )

    # 低信心：拆成「判對」與「錯非 blank」
    uc_correct_cnt = ud.get(gt_core, 0) if gt_core and gt_core not in BLANK_LABELS else 0
    uc_wrong_nonblank_cnt = sum(
        v for k, v in ud.items()
        if k not in BLANK_LABELS and k != gt_core
    )
    unclassified_nonblank = uc_wrong_nonblank_cnt

    # 這張圖的錯誤細節 + blank_miss 正確票數
    vote_err_counter = Counter()
    bm_correct_cnt = 0

    # ===== 展開成 y_true_vote / y_pred_vote =====

    # (a) 高信心正確：全部是真正「投對票」
    if correct_hi > 0:
        y_true_vote.extend([true_for_votes] * correct_hi)
        # 對非 blank 類別，true_for_votes == gt_core；對 blank 類別，true_for_votes 是 BLANK_GT_LABEL
        y_pred_vote.extend([gt_core] * correct_hi if gt_core else [true_for_votes] * correct_hi)

    # (b) blank_miss_detail：在投票視角下也要拿來比對 GT，而不是一律當「真實空白」
    for k, v in bmd.items():
        if k == BLANK_LOWCONF_KEY or v <= 0:
            continue
        if k in BLANK_LABELS:
            # 預測 blank：不算投票
            continue

        if true_for_votes == BLANK_GT_LABEL:
            # 這張圖真的是空白 → 任何非 blank 類別都是錯誤票
            vote_err_counter[k] += v
            y_true_vote.extend([BLANK_GT_LABEL] * v)
            y_pred_vote.extend([k] * v)
        else:
            # 一般情況：GT = gt_core
            if k == gt_core:
                # 像 Kl_seg.png 裡的 Kl:18 就會走到這裡 → 正確票
                bm_correct_cnt += v
                y_true_vote.extend([gt_core] * v)
                y_pred_vote.extend([gt_core] * v)
            else:
                # 例如 Kl_seg.png 裡的 bg:1 → 錯誤票
                vote_err_counter[k] += v
                y_true_vote.extend([gt_core] * v)
                y_pred_vote.extend([k] * v)

    # (c) 高信心錯（wrong_detail）：pred = 非 blank，一律當錯
    for k, v in wd.items():
        if v <= 0:
            continue
        if k in BLANK_LABELS:
            # 預測 blank：不投票
            continue
        vote_err_counter[k] += v
        y_true_vote.extend([true_for_votes] * v)
        y_pred_vote.extend([k] * v)

    # (d) 低信心（unclassified_detail）
    for k, v in ud.items():
        if v <= 0:
            continue
        if k in BLANK_LABELS:
            # 低信心 + 預測 blank：沒票
            continue

        if true_for_votes == BLANK_GT_LABEL:
            # 真實是空白 → 全部當錯誤票
            vote_err_counter[k] += v
            y_true_vote.extend([BLANK_GT_LABEL] * v)
            y_pred_vote.extend([k] * v)
        else:
            if k == gt_core:
                # 低信心但判對
                y_true_vote.extend([gt_core] * v)
                y_pred_vote.extend([gt_core] * v)
            else:
                vote_err_counter[k] += v
                y_true_vote.extend([gt_core] * v)
                y_pred_vote.extend([k] * v)

    # ===== 單張圖：統計欄位 =====
    vote_total = (
        correct_hi
        + blank_miss_nonblank
        + wrong_nonblank
        + uc_correct_cnt
        + uc_wrong_nonblank_cnt
    )
    vote_correct = correct_hi + bm_correct_cnt + uc_correct_cnt
    vote_err = max(0, vote_total - vote_correct)

    df_img.at[idx, "blank_miss_nonblank"]   = blank_miss_nonblank
    df_img.at[idx, "correct"]               = correct_hi
    df_img.at[idx, "wrong_nonblank"]        = wrong_nonblank
    df_img.at[idx, "unclassified_nonblank"] = unclassified_nonblank
    df_img.at[idx, "vote_total"]            = vote_total
    df_img.at[idx, "vote_correct"]          = vote_correct
    df_img.at[idx, "vote_err"]              = vote_err

    if vote_total > 0:
        vote_cls_acc = vote_correct / vote_total
        df_img.at[idx, "vote_cls_acc"] = round(vote_cls_acc, ROUND_NDIGITS)
        df_img.at[idx, "vote_cls_f1"]  = round(vote_cls_acc, ROUND_NDIGITS)
    else:
        df_img.at[idx, "vote_cls_acc"] = 0.0
        df_img.at[idx, "vote_cls_f1"]  = 0.0

    # 🔴 先把這張圖的錯誤累積到「全體錯誤 Counter」
    global_vote_err_counter.update(vote_err_counter)

    # 錯誤細節字串（只記「真的錯」的那幾類）
    if vote_err_counter:
        df_img.at[idx, "vote_err_detail"] = "; ".join(
            f"{k}:{v}" for k, v in sorted(vote_err_counter.items())
        )
    else:
        df_img.at[idx, "vote_err_detail"] = ""

# ===== 用 sklearn 算整體指標（投票視角 + 含 blank_miss_detail） =====
if len(y_true_vote) > 0:
    overall_acc = accuracy_score(y_true_vote, y_pred_vote)
    micro_f1    = f1_score(y_true_vote, y_pred_vote, average="micro")
    if str(target_group) == "BGC":
        y_true_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in y_true_vote]
        y_pred_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in y_pred_vote]
        macro_f1   = f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0)
    else:
        macro_f1   = f1_score(y_true_vote, y_pred_vote, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_vote, y_pred_vote, average="weighted")
else:
    overall_acc = micro_f1 = macro_f1 = weighted_f1 = 0.0  # 或 np.nan 也可以


# ====== 建立 TOTAL 行（整體數量，非 detail） ======
total_row = df_img[sum_cols].sum(numeric_only=True)

# 總體的 acc / f1：用「總正確 / 總票數」
vt = float(total_row.get("vote_total", 0))
vc = float(total_row.get("vote_correct", 0))

if vt > 0:
    total_row["vote_cls_acc"] = round(vc / vt, ROUND_NDIGITS)
    total_row["vote_cls_f1"]  = total_row["vote_cls_acc"]
else:
    total_row["vote_cls_acc"] = 0.0
    total_row["vote_cls_f1"]  = 0.0

# 用 sklearn 算出來的「整體」指標，寫進 TOTAL 那一列
total_row["overall_acc"] = round(overall_acc, ROUND_NDIGITS)
total_row["micro_f1"]    = round(micro_f1, ROUND_NDIGITS)
total_row["macro_f1"]    = round(macro_f1, ROUND_NDIGITS)
total_row["weighted_f1"] = round(weighted_f1, ROUND_NDIGITS)

# 🔴 TOTAL 的 vote_err_detail：全圖錯誤最多的前 K 類
if global_vote_err_counter:
    top_items = global_vote_err_counter.most_common(TOP_K)
    total_row["vote_err_detail"] = "; ".join(f"{k}:{v}" for k, v in top_items)
else:
    total_row["vote_err_detail"] = ""

# 顯示用的名稱；你要叫 "TOTAL" 或 "TOTAL_vote" 都可以
total_row["img"] = TOTAL_NAME

# 組成一個新的 DataFrame 來印（避免影響後面 merge）
df_print = pd.concat([df_img, pd.DataFrame([total_row])], ignore_index=True)

print("=== 每張圖（SAM+分類循環的投票視角） ===")
print(df_print[PRINT_COLS].to_string(index=False))

print("\n=== Overall（投票視角，sklearn + 含 blank_miss_detail） ===")
print(f"overall_acc : {overall_acc:.4f}")
print(f"micro_f1    : {micro_f1:.4f}")
print(f"macro_f1    : {macro_f1:.4f}")
print(f"weighted_f1 : {weighted_f1:.4f}")

# ====== 存成 vote 視角專用 CSV（欄位就跟你 print 的一樣） ======
df_print[cols_to_export].to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n已輸出含投票欄位的 CSV：{out_csv}")

