import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import json, cv2, tqdm

import sys
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
'''
# =========================
# 0) 可調參數
# =========================
# 你要整合的報告 csv 路徑（可手動列出多個）
REPORT_CSVS = [
    f"/data/ch21908234/work/SAM_Classifier_Feedback_Loop/reports/{method}/color correction_XIV/fixed_train_data_20250916_015027/cls_report_image_view_white_le50.csv",
    f"/data/ch21908234/work/SAM_Classifier_Feedback_Loop/reports/{method}/color correction_XIII/fixed_train_data_20250916_015027/cls_report_image_view_white_le50.csv",
]

# 若你希望 gt_core 也走你原本的「單類→複合提升」邏輯（Mk -> Mag-Mk）
# 請填入 train_data.json（或你用來建 canonical mapping 的 json）
JSON_PATH_FOR_CANON = None  # e.g. "/data/.../train_data.json"

# 合併後輸出
OUT_MERGED_CSV = f"/data/ch21908234/work/SAM_Classifier_Feedback_Loop/reports/{method}/merged_reports_color correction_XIV_XIII_cls_report_image_view_white_le50.csv"
'''

from pathlib import Path

#在1930這個日治地圖上預設

scale  = "all"  # 可輸入 "5w" / "10w" / "all"
method = config.args.cls_method
target_group = config.args.cls_eval_target.replace('legends', 'FG').replace('others', 'BGC')  #當前檢測項目，BGC、FG
# =========================
# 0) 可調參數（自動 discover）
# =========================
ROOT = Path(f"reports/{method}/{target_group}")

REPORT_FILENAME = "cls_report_image_view_white_le50.csv"

# 若你希望 gt_core 也走你原本的「單類→複合提升」邏輯（Mk -> Mag-Mk）
JSON_PATH_FOR_CANON = None  # e.g. "/data/.../train_data.json"


# =========================
# discover helpers
# =========================
def _is_5w_map(map_name: str) -> bool:
    # 5w 地圖名必定是數字：17、13...
    return str(map_name).isdigit()

def _is_10w_map(map_name: str) -> bool:
    # 10w 地圖名必定是英文：XIII、XIV...
    s = str(map_name).strip()
    return s.isalpha() and s.upper() == s

def _parse_stage_and_map(folder_name: str):
    """
    第二層資料夾可能是：
      - "color correction_13"  -> stage="color correction", map="13"
      - "color correction_XIV" -> stage="color correction", map="XIV"
      - "13"                   -> stage=None, map="13"
    """
    s = str(folder_name).strip()
    if "_" in s:
        left, right = s.rsplit("_", 1)
        right = right.strip()
        if right.isdigit() or _is_10w_map(right):
            stage = left.strip()
            map_name = right
            return stage, map_name
    return None, s  # 只有地圖名

def _scale_keep(map_name: str, scale: str) -> bool:
    if scale == "all":
        return True
    if scale == "5w":
        return _is_5w_map(map_name)
    if scale == "10w":
        return _is_10w_map(map_name)
    raise ValueError(f"scale 只能是 5w/10w/all，但你給的是：{scale}")


# =========================
# 產生 REPORT_CSVS
# =========================
all_found = sorted(ROOT.rglob(REPORT_FILENAME))

rows = []  # (stage, map_name, path, stage_map_dirname)
for p in all_found:
    # 期待結構：.../<第二層>/fixed_train_data_xxx/<report.csv>
    try:
        stage_map_dir = p.parents[1]  # fixed_train_data_xxx 的上一層
        stage_map_name = stage_map_dir.name
    except Exception:
        stage_map_name = None

    stage, map_name = _parse_stage_and_map(stage_map_name or "UNKNOWN")

    # scale 篩選（all / 5w / 10w）
    if not _scale_keep(map_name, scale):
        continue
    assert all([item[1]!=map_name for item in rows]), f'{map_name} map has multiple results! remove one of them before compute the overall metrics. {p} vs {rows}'
    rows.append((stage, map_name, p, stage_map_name))

# 最終要丟進 merge 的 csv list
REPORT_CSVS = [str(p) for (_, _, p, _) in rows]


# =========================
# OUT_MERGED_CSV 命名規則（修正版）
# =========================
if scale == "all":
    # 你要的：merged_reports_all_cls_report_image_view_white_le50.csv
    OUT_MERGED_CSV = str(ROOT / f"merged_reports_all_{REPORT_FILENAME}")
else:
    # 非 all 時：維持原本邏輯（用 stage 或 mixed + maps）
    stages = sorted({s for (s, _, _, _) in rows if s})

    if len(stages) == 0:
        var_tag = "orig"         # 你說的「沒寫 stage 就叫預設」
    elif len(stages) == 1:
        var_tag = stages[0]      # 例如 "color correction"
    else:
        var_tag = "multi_stage"  # 多個 stage 混在一起（不要誤叫 orig）


    maps = sorted({m for (_, m, _, _) in rows}, key=lambda x: (len(str(x)), str(x)))
    maps_tag = "_".join(maps) if maps else "NONE"

    OUT_MERGED_CSV = str(ROOT / f"merged_reports_{var_tag}_{maps_tag}_{REPORT_FILENAME}")



# =========================
# 1) canonical mapping（可選，模仿你原本的做法）
# =========================
_COMPONENTS_SET_TO_CANON = {}
_PART_TO_BEST_CANON = {}

def _clean_base_token(tok: str) -> str:
    if tok is None:
        return None
    s = str(tok).strip()
    s = s.split('_', 1)[0]
    m = re.match(r'^([A-Za-z]+)', s)
    return m.group(1) if m else s

def _extract_labels_from_json_file(path: str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            jd = json.load(f)

        classes_raw = []
        if isinstance(jd, dict):
            if "classes" in jd and isinstance(jd["classes"], dict):
                classes_raw = list(jd["classes"].keys())
            elif "classes" in jd and isinstance(jd["classes"], (list, tuple)):
                classes_raw = list(jd["classes"])
            else:
                for k in ["class_names", "labels", "label_names"]:
                    if k in jd and isinstance(jd[k], (list, tuple)):
                        classes_raw = list(jd[k])
                        break
        elif isinstance(jd, list):
            classes_raw = list(jd)

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

def build_canon_maps_from_json(json_path: str):
    _COMPONENTS_SET_TO_CANON.clear()
    _PART_TO_BEST_CANON.clear()

    raw_labels = _extract_labels_from_json_file(json_path)
    if not raw_labels:
        print("[warn] JSON 抓不到 classes，canonical mapping 會退化成純清洗 token。")
        return

    part_to_cands = {}
    for lbl in raw_labels:
        parts = lbl.split('-')
        if len(parts) >= 2:
            _COMPONENTS_SET_TO_CANON.setdefault(frozenset(parts), lbl)
            for p in parts:
                part_to_cands.setdefault(p, []).append(lbl)

    def _comp_len(x: str) -> int:
        return len(x.split('-'))

    for p, cands in part_to_cands.items():
        # 子項數最多優先，其次依 raw_labels 先後
        cands_sorted = sorted(cands, key=lambda x: (-_comp_len(x), raw_labels.index(x)))
        _PART_TO_BEST_CANON[p] = cands_sorted[0]

def normalize_core(name: str) -> str:
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
        return _PART_TO_BEST_CANON.get(p, p)  # 單類→複合提升（若 mapping 有）
    canon = _COMPONENTS_SET_TO_CANON.get(frozenset(parts))
    return canon if canon else '-'.join(parts)

def core_from_img_name(img_name: str) -> str:
    # 你的報告裡 img 通常是 "Mk_001.png" 這種
    stem = Path(str(img_name)).stem
    # 先砍掉 "_後綴"
    base = stem.split('_', 1)[0]
    return normalize_core(base)

if JSON_PATH_FOR_CANON:
    build_canon_maps_from_json(JSON_PATH_FOR_CANON)


# =========================
# 2) 解析 wrong_detail / unclassified_detail
# =========================
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
import json

# ---- 解析 detail 字串 ----
def parse_detail(s) -> dict:
    if s is None:
        return {}
    if isinstance(s, float) and np.isnan(s):
        return {}
    s = str(s).strip()
    if not s:
        return {}
    out = {}
    for seg in s.split(";"):
        seg = seg.strip()
        if not seg or ":" not in seg:
            continue
        k, v = seg.split(":", 1)
        k = str(k).strip()
        v = str(v).strip()
        if not k:
            continue
        try:
            out[k] = out.get(k, 0) + int(float(v))
        except:
            pass
    return out

def format_detail(counter: Counter) -> str:
    # 依 count 由大到小輸出
    if not counter:
        return ""
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return "; ".join(f"{k}:{v}" for k, v in items)

# =========================
# (可選) canonical mapping：如果你已經在前面建好 normalize_core/core_from_img_name，這裡就直接用
# 這裡假設你「已有」 normalize_core(...) 與 core_from_img_name(...)
# =========================

def merge_reports_and_compute(report_csvs, out_csv="merged_reports.csv"):
    all_rows = []
    global_y_true, global_y_pred, global_w = [], [], []

    per_report_summary = []

    for csv_path in report_csvs:
        csv_path = str(csv_path)
        src = str(Path(csv_path))            # 你傳的是 /data/... 就會 그대로 /data/...
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        if "img" not in df.columns:
            raise ValueError(f"[{src}] 缺少 img 欄位，這份不是 patch_stats 格式？")

        # 只取非 TOTAL 的逐圖 rows
        df2 = df[df["img"].astype(str) != "TOTAL"].copy()

        # 欄位容錯
        def pick_col(*cands):
            for c in cands:
                if c in df2.columns:
                    return c
            return None

        col_correct = pick_col("correct")
        col_wrong   = pick_col("wrong")
        col_uncls   = pick_col("unclassified")
        col_wrong_detail = pick_col("wrong_detail")
        col_uncls_detail = pick_col("unclassified_detail")

        missing = [k for k,v in {
            "correct": col_correct, "wrong": col_wrong, "unclassified": col_uncls,
            "wrong_detail": col_wrong_detail, "unclassified_detail": col_uncls_detail
        }.items() if v is None]
        if missing:
            raise ValueError(f"[{src}] 缺少欄位：{missing}，你這份報告版本不一致。")

        # build weighted samples（用 sample_weight，不展開）
        y_t, y_p, w = [], [], []
        for _, r in df2.iterrows():
            gt = core_from_img_name(r["img"])

            correct = int(float(r[col_correct])) if pd.notna(r[col_correct]) else 0
            wrong   = int(float(r[col_wrong]))   if pd.notna(r[col_wrong])   else 0
            uncls   = int(float(r[col_uncls]))   if pd.notna(r[col_uncls])   else 0

            if correct > 0:
                y_t.append(gt); y_p.append(gt); w.append(correct)

            wrong_map = parse_detail(r[col_wrong_detail])
            for pred, cnt in wrong_map.items():
                pred = normalize_core(pred)
                if cnt > 0:
                    y_t.append(gt); y_p.append(pred); w.append(cnt)

            uncls_map = parse_detail(r[col_uncls_detail])
            for pred, cnt in uncls_map.items():
                pred = normalize_core(pred)
                if cnt > 0:
                    y_t.append(gt); y_p.append(pred); w.append(cnt)

        # per-report metrics
        acc = accuracy_score(y_t, y_p, sample_weight=w) if len(w) else 0.0
        micro = f1_score(y_t, y_p, average="micro", sample_weight=w, zero_division=0) if len(w) else 0.0
        if len(w):
            if str(target_group) == "BGC":
                y_true_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in y_t]
                y_pred_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in y_p]
                macro = f1_score(
                    y_true_bin, y_pred_bin,
                    average="binary", pos_label=1,
                    sample_weight=w, zero_division=0
                )
            else:
                macro = f1_score(
                    y_t, y_p,
                    average="macro",
                    sample_weight=w, zero_division=0
                )
        else:
            macro = 0.0        
        weighted = f1_score(y_t, y_p, average="weighted", sample_weight=w, zero_division=0) if len(w) else 0.0

        per_report_summary.append({
            "source": src,
            "overall_acc": round(acc, 4),
            "micro_f1": round(micro, 4),
            "macro_f1": round(macro, 4),
            "weighted_f1": round(weighted, 4),
            "n_weighted_samples": int(sum(w)) if len(w) else 0,
        })

        # 合併逐圖 rows（加上來源）
        df2.insert(0, "source", src)
        all_rows.append(df2)

        # 併入 global sample
        global_y_true.extend(y_t)
        global_y_pred.extend(y_p)
        global_w.extend(w)

    merged_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(per_report_summary)

    # ---- global overall metrics (sklearn) ----
    if len(global_w):
        overall_acc = accuracy_score(global_y_true, global_y_pred, sample_weight=global_w)
        micro_f1    = f1_score(global_y_true, global_y_pred, average="micro", sample_weight=global_w, zero_division=0)
        if str(target_group) == "BGC":
            y_true_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in global_y_true]
            y_pred_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in global_y_pred]
            macro_f1   = f1_score(y_true_bin, y_pred_bin, average="binary",
                                  pos_label=1, sample_weight=global_w, zero_division=0)
        else:
            macro_f1   = f1_score(global_y_true, global_y_pred, average="macro",
                                  sample_weight=global_w, zero_division=0)
        weighted_f1 = f1_score(global_y_true, global_y_pred, average="weighted", sample_weight=global_w, zero_division=0)
    else:
        overall_acc = micro_f1 = macro_f1 = weighted_f1 = 0.0

    # ---- 建 GLOBAL TOTAL（完整）----
    if not merged_df.empty:
        # 確保欄位存在（沒有的就先補）
        for c in ["overall_acc","micro_f1","macro_f1","weighted_f1"]:
            if c not in merged_df.columns:
                merged_df[c] = np.nan

        # 你原本 TOTAL 會加總的欄位（存在才加）
        sum_cols = [
            "total","checked","blank_area","content_area",
            "blank(pred)","blank_miss","correct","wrong","unclassified"
        ]
        sum_cols = [c for c in sum_cols if c in merged_df.columns]

        total_row = {c: "" for c in merged_df.columns}
        total_row["source"] = "GLOBAL"
        total_row["img"] = "TOTAL"

        # 1) 數字欄位加總
        for c in sum_cols:
            total_row[c] = pd.to_numeric(merged_df[c], errors="coerce").fillna(0).sum()

        # 2) quality_all / quality_content：用 checked 加權平均
        if "checked" in merged_df.columns:
            checked_sum = pd.to_numeric(merged_df["checked"], errors="coerce").fillna(0).sum()
            if checked_sum > 0:
                if "quality_all" in merged_df.columns:
                    qa = pd.to_numeric(merged_df["quality_all"], errors="coerce").fillna(0)
                    ch = pd.to_numeric(merged_df["checked"], errors="coerce").fillna(0)
                    total_row["quality_all"] = float((qa * ch).sum() / checked_sum)
                if "quality_content" in merged_df.columns:
                    qc = pd.to_numeric(merged_df["quality_content"], errors="coerce").fillna(0)
                    ch = pd.to_numeric(merged_df["checked"], errors="coerce").fillna(0)
                    total_row["quality_content"] = float((qc * ch).sum() / checked_sum)

        # 3) cls_acc / cls_f1：用 content_area 加權平均
        if "content_area" in merged_df.columns:
            content_sum = pd.to_numeric(merged_df["content_area"], errors="coerce").fillna(0).sum()
            if content_sum > 0:
                if "cls_acc" in merged_df.columns:
                    ca = pd.to_numeric(merged_df["cls_acc"], errors="coerce").fillna(0)
                    wgt = pd.to_numeric(merged_df["content_area"], errors="coerce").fillna(0)
                    total_row["cls_acc"] = float((ca * wgt).sum() / content_sum)
                if "cls_f1" in merged_df.columns:
                    cf = pd.to_numeric(merged_df["cls_f1"], errors="coerce").fillna(0)
                    wgt = pd.to_numeric(merged_df["content_area"], errors="coerce").fillna(0)
                    total_row["cls_f1"] = float((cf * wgt).sum() / content_sum)

        # 4) detail 欄位累加（若存在）
        agg_wrong = Counter()
        agg_uncls = Counter()
        agg_blank_miss = Counter()

        if "wrong_detail" in merged_df.columns:
            for s in merged_df["wrong_detail"].tolist():
                agg_wrong.update(parse_detail(s))
            total_row["wrong_detail"] = format_detail(agg_wrong)

        if "unclassified_detail" in merged_df.columns:
            for s in merged_df["unclassified_detail"].tolist():
                agg_uncls.update(parse_detail(s))
            total_row["unclassified_detail"] = format_detail(agg_uncls)

        if "blank_miss_detail" in merged_df.columns:
            for s in merged_df["blank_miss_detail"].tolist():
                agg_blank_miss.update(parse_detail(s))
            total_row["blank_miss_detail"] = format_detail(agg_blank_miss)

        # 5) 塞 sklearn overall 4 指標
        total_row["overall_acc"] = round(float(overall_acc), 4)
        total_row["micro_f1"]    = round(float(micro_f1), 4)
        total_row["macro_f1"]    = round(float(macro_f1), 4)
        total_row["weighted_f1"] = round(float(weighted_f1), 4)

        # （可選）把一些數字欄位 round 一下更像你原本輸出
        for c in ["quality_all","quality_content","cls_acc","cls_f1"]:
            if c in total_row and total_row[c] != "":
                try:
                    total_row[c] = round(float(total_row[c]), 4)
                except:
                    pass

        merged_df.loc[len(merged_df)] = total_row
        merged_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("=== Per-report Summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Global Overall (tolerant from merged reports) ===")
    print(f"Overall Accuracy : {overall_acc:.4f}")
    print(f"Micro-F1         : {micro_f1:.4f}")
    print(f"Macro-F1         : {macro_f1:.4f}")
    print(f"Weighted-F1      : {weighted_f1:.4f}")

    if not merged_df.empty:
        print(f"\nMerged CSV saved: {out_csv}")

    return merged_df, summary_df, (overall_acc, micro_f1, macro_f1, weighted_f1)

# =========================
# 5) 執行
# =========================

if len(REPORT_CSVS) == 0:
    print("請先在 REPORT_CSVS 填入你要整合的多個 patch_stats*.csv 路徑")
else:
    merged_df, summary_df, global_metrics = merge_reports_and_compute(REPORT_CSVS, out_csv=OUT_MERGED_CSV)





################################################
JSON_PATH = None  # e.g. "/content/train_data.json"

BLANK_LABELS = {"blank"}
BLANK_GT_LABEL = "__blank_gt__"          # 跟你舊版一致
UNKNOWN_ERR_LABEL = "__err_unknown__"    # vote_err_detail 不足時補洞用
TOP_K = 5                                 # TOTAL 的 vote_err_detail 顯示前 K 名


from pathlib import Path

#在1930這個日治地圖上預設

scale  = "all"  # 可輸入 "5w" / "10w" / "all"

# =========================
# 0) 可調參數（自動 discover）
# =========================
ROOT = Path(f"reports/{method}/{target_group}")

REPORT_FILENAME = "cls_report_vote_view.csv"

# 若你希望 gt_core 也走你原本的「單類→複合提升」邏輯（Mk -> Mag-Mk）
JSON_PATH_FOR_CANON = None  # e.g. "/data/.../train_data.json"


# =========================
# discover helpers
# =========================
def _is_5w_map(map_name: str) -> bool:
    # 5w 地圖名必定是數字：17、13...
    return str(map_name).isdigit()

def _is_10w_map(map_name: str) -> bool:
    # 10w 地圖名必定是英文：XIII、XIV...
    s = str(map_name).strip()
    return s.isalpha() and s.upper() == s

def _parse_stage_and_map(folder_name: str):
    """
    第二層資料夾可能是：
      - "color correction_13"  -> stage="color correction", map="13"
      - "color correction_XIV" -> stage="color correction", map="XIV"
      - "13"                   -> stage=None, map="13"
    """
    s = str(folder_name).strip()
    if "_" in s:
        left, right = s.rsplit("_", 1)
        right = right.strip()
        if right.isdigit() or _is_10w_map(right):
            stage = left.strip()
            map_name = right
            return stage, map_name
    return None, s  # 只有地圖名

def _scale_keep(map_name: str, scale: str) -> bool:
    if scale == "all":
        return True
    if scale == "5w":
        return _is_5w_map(map_name)
    if scale == "10w":
        return _is_10w_map(map_name)
    raise ValueError(f"scale 只能是 5w/10w/all，但你給的是：{scale}")


# =========================
# 產生 REPORT_CSVS
# =========================
all_found = sorted(ROOT.rglob(REPORT_FILENAME))

rows = []  # (stage, map_name, path, stage_map_dirname)
for p in all_found:
    # 期待結構：.../<第二層>/fixed_train_data_xxx/<report.csv>
    try:
        stage_map_dir = p.parents[1]  # fixed_train_data_xxx 的上一層
        stage_map_name = stage_map_dir.name
    except Exception:
        stage_map_name = None

    stage, map_name = _parse_stage_and_map(stage_map_name or "UNKNOWN")

    # scale 篩選（all / 5w / 10w）
    if not _scale_keep(map_name, scale):
        continue
    assert all([item[1]!=map_name for item in rows]), f'{map_name} map has multiple results! remove one of them before compute the overall metrics. {p} vs {rows}'
    rows.append((stage, map_name, p, stage_map_name))

# 最終要丟進 merge 的 csv list
REPORT_CSVS = [str(p) for (_, _, p, _) in rows]


# =========================
# OUT_MERGED_CSV 命名規則（修正版）
# =========================
if scale == "all":
    # 你要的：merged_reports_all_cls_report_image_view_white_le50.csv
    OUT_MERGED_CSV = str(ROOT / f"merged_reports_all_{REPORT_FILENAME}")
else:
    # 非 all 時：維持原本邏輯（用 stage 或 mixed + maps）
    stages = sorted({s for (s, _, _, _) in rows if s})

    if len(stages) == 0:
        var_tag = "orig"         # 你說的「沒寫 stage 就叫預設」
    elif len(stages) == 1:
        var_tag = stages[0]      # 例如 "color correction"
    else:
        var_tag = "multi_stage"  # 多個 stage 混在一起（不要誤叫 orig）


    maps = sorted({m for (_, m, _, _) in rows}, key=lambda x: (len(str(x)), str(x)))
    maps_tag = "_".join(maps) if maps else "NONE"

    OUT_MERGED_CSV = str(ROOT / f"merged_reports_{var_tag}_{maps_tag}_{REPORT_FILENAME}")


# 讓下面舊版變數名也能吃到自動 discover 的結果
SRC_CSVS = REPORT_CSVS
OUT_CSV  = OUT_MERGED_CSV


# =========================
# 1) label normalize（跟你舊版一致 + 可選 JSON 升級）
# =========================
_COMPONENTS_SET_TO_CANON = {}
_PART_TO_BEST_CANON = {}

def _clean_base_token(tok: str) -> str:
    """子 token 僅保留開頭英文字母；支援 Mk_001、Mag002、Qo-1 等寫法。"""
    if tok is None:
        return None
    s = str(tok).strip()
    s = s.split('_', 1)[0]
    m = re.match(r'^([A-Za-z]+)', s)
    return m.group(1) if m else s

def extract_class_list_from_json(jobj) -> list:
    """僅清洗、不合併；保留 '-' 結構。"""
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

def _comp_len(x: str) -> int:
    return len(x.split('-'))

def build_composite_mapping_from_json(json_path: str):
    """建立 Mk -> Mag-Mk 這種升級映射（跟你舊版一樣）。"""
    _COMPONENTS_SET_TO_CANON.clear()
    _PART_TO_BEST_CANON.clear()
    if not json_path:
        return
    jp = Path(json_path)
    if not jp.exists():
        print(f"[warn] JSON_PATH 不存在：{jp}（將只做基本清洗，不做複合升級）")
        return

    with open(jp, "r", encoding="utf-8") as f:
        jd = json.load(f)
    labels = extract_class_list_from_json(jd)

    _part_to_cands = {}
    for lbl in labels:
        parts = lbl.split('-')
        if len(parts) >= 2:
            key = frozenset(parts)
            _COMPONENTS_SET_TO_CANON.setdefault(key, lbl)
            for p in parts:
                _part_to_cands.setdefault(p, []).append(lbl)

    for p, cands in _part_to_cands.items():
        cands_sorted = sorted(cands, key=lambda x: (-_comp_len(x), labels.index(x)))
        _PART_TO_BEST_CANON[p] = cands_sorted[0]

def normalize_core(name: str) -> str:
    """
    正規化（含可選複合升級）：
    - 單一子項：若可映射到複合 → 提升成複合
    - 多子項：依 parts 集合查 _COMPONENTS_SET_TO_CANON（順序無關）
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

    # 沒有 JSON 映射時，就只做基本清洗
    has_map = bool(_PART_TO_BEST_CANON) or bool(_COMPONENTS_SET_TO_CANON)

    if len(parts) == 1:
        p = parts[0]
        return _PART_TO_BEST_CANON.get(p, p) if has_map else p

    if has_map:
        canon = _COMPONENTS_SET_TO_CANON.get(frozenset(parts))
        return canon if canon else '-'.join(parts)
    else:
        return '-'.join(parts)

def core_of(img_name: str) -> str:
    """從 img 欄位萃取 GT core（跟你舊版一致：stem -> normalize）。"""
    if not img_name:
        return None
    base = Path(str(img_name)).stem
    base = str(base).split('_', 1)[0]
    return normalize_core(base)

def parse_vote_err_detail(s) -> Counter:
    """
    解析 vote_err_detail：例如 'Oku:3; Qy:2'
    回傳 Counter(pred_label -> count)，並做 normalize_core。
    """
    c = Counter()
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return c
    ss = str(s).strip()
    if not ss:
        return c
    ss = ss.replace("：", ":")
    for tok in re.split(r"[;,|\n]+", ss):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            k, v = tok.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                n = int(round(float(v)))
            except ValueError:
                continue
        else:
            k = tok
            n = 1
        if not k or n <= 0:
            continue
        # vote_err_detail 基本上都是類別名，但我們仍做一次 normalize
        kk = normalize_core(k)
        if kk:
            c[kk] += n
    return c

# 先建 mapping（可選）
build_composite_mapping_from_json(JSON_PATH)

# =========================
# 2) 讀多份 vote_view.csv，重建「加權樣本」算總體指標
# =========================
if not SRC_CSVS:
    raise ValueError("SRC_CSVS 是空的。請先把 patch_stats_vote_view.csv 路徑填進去。")

dfs = []
pair_counter = Counter()       # (true_label, pred_label) -> count（用 sample_weight 壓縮，不展開每一票）
global_err_pred_counter = Counter()

sum_cols = [
    "blank_miss_nonblank",
    "correct",
    "wrong_nonblank",
    "unclassified_nonblank",
    "vote_total",
    "vote_correct",
    "vote_err",
]

for src in SRC_CSVS:
    src = str(src)
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"找不到：{p}")

    df = pd.read_csv(p)
    if "img" not in df.columns:
        raise ValueError(f"{p} 缺少欄位 img")

    # 只取非 TOTAL
    is_total = df["img"].astype(str).str.upper().eq("TOTAL")
    df_img = df[~is_total].copy()

    # 基本欄位檢查
    need_cols = {"vote_total", "vote_correct", "vote_err", "vote_err_detail"}
    miss = [c for c in need_cols if c not in df_img.columns]
    if miss:
        raise ValueError(f"{p} 缺少欄位：{miss}（你輸入的必須是 patch_stats_vote_view.csv）")

    df_img.insert(0, "source_csv", src)
    dfs.append(df_img)

    # 累積 (y_true, y_pred, weight)
    for _, row in df_img.iterrows():
        img_name = row.get("img", "")
        gt_core = core_of(img_name)

        true_label = gt_core if (gt_core and gt_core not in BLANK_LABELS) else BLANK_GT_LABEL
        pred_label_correct = gt_core if gt_core else true_label  # 跟你舊版「correct_hi」邏輯一致

        # 票數
        vt = int(row.get("vote_total", 0) or 0)
        vc = int(row.get("vote_correct", 0) or 0)
        ve = int(row.get("vote_err", 0) or 0)

        # 容錯：如果 vote_total 不等於 correct+err，用 vote_total 補（以免少算）
        if vt <= 0:
            # 沒票就跳過（不影響 sklearn 指標）
            continue

        # correct
        if vc > 0:
            pair_counter[(true_label, pred_label_correct)] += vc

        # err detail
        err_c = parse_vote_err_detail(row.get("vote_err_detail", ""))
        sum_detail = sum(err_c.values())

        # 如果 detail 與 vote_err 不一致，補洞/裁切，避免 total weight 不對
        if ve > 0:
            if sum_detail < ve:
                err_c[UNKNOWN_ERR_LABEL] += (ve - sum_detail)
            elif sum_detail > ve:
                # 超過就把多的從最大的類別扣掉
                over = sum_detail - ve
                for k, _cnt in err_c.most_common():
                    if over <= 0:
                        break
                    take = min(err_c[k], over)
                    err_c[k] -= take
                    over -= take
                    if err_c[k] <= 0:
                        del err_c[k]

        # 累積錯誤票
        for pred_k, cnt in err_c.items():
            if cnt <= 0:
                continue
            pair_counter[(true_label, pred_k)] += cnt
            global_err_pred_counter[pred_k] += cnt

# 合併成一張表（每張圖列）
df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# =========================
# 3) 用 sklearn（sample_weight）算整體指標
# =========================
if not pair_counter:
    overall_acc = micro_f1 = macro_f1 = weighted_f1 = 0.0
else:
    y_true_small = []
    y_pred_small = []
    w = []
    for (t, p), cnt in pair_counter.items():
        if cnt <= 0:
            continue
        y_true_small.append(t)
        y_pred_small.append(p)
        w.append(cnt)

    overall_acc = accuracy_score(y_true_small, y_pred_small, sample_weight=w)
    micro_f1    = f1_score(y_true_small, y_pred_small, average="micro", sample_weight=w)
    if str(target_group) == "BGC":
        y_true_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in y_true_small]
        y_pred_bin = [1 if (y == "BGC" or y == "bg") else 0 for y in y_pred_small]
        macro_f1   = f1_score(y_true_bin, y_pred_bin, average="binary",
                              pos_label=1, sample_weight=w, zero_division=0)
    else:
        macro_f1   = f1_score(y_true_small, y_pred_small, average="macro",
                              sample_weight=w, zero_division=0)
    weighted_f1 = f1_score(y_true_small, y_pred_small, average="weighted", sample_weight=w)

print("\n=== Overall（合併所有 patch_stats_vote_view.csv 的投票視角；sklearn 計算） ===")
print(f"Overall Accuracy : {overall_acc:.4f}")
print(f"Micro-F1 : {micro_f1:.4f}")
print(f"Macro-F1 : {macro_f1:.4f}")
print(f"Weighted-F1 : {weighted_f1:.4f}")

# =========================
# 4) 產生最後的 TOTAL row + 輸出合併 CSV
# =========================
# 如果 df_all 沒有某些 sum_cols，就自動補 0
for c in sum_cols:
    if c not in df_all.columns:
        df_all[c] = 0

total_row = df_all[sum_cols].sum(numeric_only=True)

vt = float(total_row.get("vote_total", 0))
vc = float(total_row.get("vote_correct", 0))
if vt > 0:
    total_row["vote_cls_acc"] = round(vc / vt, 4)
    total_row["vote_cls_f1"]  = total_row["vote_cls_acc"]
else:
    total_row["vote_cls_acc"] = 0.0
    total_row["vote_cls_f1"]  = 0.0

total_row["overall_acc"] = round(overall_acc, 4)
total_row["micro_f1"]    = round(micro_f1, 4)
total_row["macro_f1"]    = round(macro_f1, 4)
total_row["weighted_f1"] = round(weighted_f1, 4)

# TOTAL 的 vote_err_detail：全體錯誤最多的前 TOP_K 類
if global_err_pred_counter:
    top_items = global_err_pred_counter.most_common(TOP_K)
    total_row["vote_err_detail"] = "; ".join(f"{k}:{v}" for k, v in top_items)
else:
    total_row["vote_err_detail"] = ""

total_row["img"] = "TOTAL"
total_row["source_csv"] = "MERGED"

# 你原本的輸出欄位 + 我加的 source_csv
cols_to_export = [
    "source_csv",
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

# 補齊缺欄
for c in cols_to_export:
    if c not in df_all.columns:
        df_all[c] = np.nan

df_out = pd.concat([df_all[cols_to_export], pd.DataFrame([total_row])[cols_to_export]], ignore_index=True)
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n已輸出合併 CSV：{OUT_CSV}")

################################
# 不要在 others/BGC 時 exit，否則 BGC 跑完後不會更新 summary_metrics_4cols.csv
# if config.args.cls_eval_target!='legends':
#     exit()

# 只畫當前 group，避免每次都重掃整個 reports
IN_DIR = Path(f"reports/{method}/{target_group}")

PATCH_H, PATCH_W = 56, 98
LINE_THICKNESS = 1
RECURSIVE = True          # True: 連子資料夾一起做；False: 只做這層
OVERWRITE = True          # True: 直接覆蓋原檔（原本圖片不用留）
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ====== 2) 收集圖片 ======
IN_DIR = IN_DIR.expanduser().resolve()
if not IN_DIR.exists():
    raise FileNotFoundError(f"資料夾不存在：{IN_DIR}")

paths = list(IN_DIR.rglob("*") if RECURSIVE else IN_DIR.glob("*"))
img_paths = [p for p in paths if p.is_file() and p.suffix.lower() in VALID_EXT]

print(f"[INFO] IN_DIR = {IN_DIR}")
print(f"[INFO] Found images: {len(img_paths)}")

# ====== 3) 批次畫網格並存回去 ======
black = (0, 0, 0)
ok, fail, skipped = 0, 0, 0

for p in tqdm.tqdm(img_paths, desc="Drawing grid"):
    img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img_bgr is None:
        fail += 1
        continue

    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()

    # 垂直線：每 PATCH_W 一條
    for x in range(0, W, PATCH_W):
        cv2.line(out, (x, 0), (x, H - 1), black, thickness=LINE_THICKNESS, lineType=cv2.LINE_8)

    # 水平線：每 PATCH_H 一條
    for y in range(0, H, PATCH_H):
        cv2.line(out, (0, y), (W - 1, y), black, thickness=LINE_THICKNESS, lineType=cv2.LINE_8)

    # 最外框
    cv2.rectangle(out, (0, 0), (W - 1, H - 1), black, thickness=LINE_THICKNESS, lineType=cv2.LINE_8)

    # 覆蓋原圖（或另存）
    if OVERWRITE:
        out_path = p
    else:
        out_path = p.with_name(p.stem + "__grid" + p.suffix)

    if not cv2.imwrite(str(out_path), out):
        fail += 1
    else:
        ok += 1

print(f"[DONE] ok={ok}, fail={fail}, skipped={skipped}")

########################

ROOT = Path(f"reports/{method}")

TARGET_NAMES = {
    "cls_report_image_view_white_le50.csv",
    "cls_report_vote_view.csv",
    "merged_reports_all_cls_report_image_view_white_le50.csv",
    "merged_reports_all_cls_report_vote_view.csv",
}

METRIC_COLS = ["overall_acc", "micro_f1", "macro_f1", "weighted_f1"]

assert ROOT.exists(), f"ROOT 不存在：{ROOT}"

def pick_metric_row(df: pd.DataFrame):
    """
    優先抓整體那一列：
    1. source_csv == 'MERGED' 或 img == 'TOTAL'
    2. 四個 metric 有值的最後一列
    3. 都沒有就回傳 None
    """
    df2 = df.copy()

    # 去空白，避免 ' MERGED ' 這種情況
    for c in ["source_csv", "img"]:
        if c in df2.columns:
            df2[c] = df2[c].astype(str).str.strip()

    # 先找 MERGED / TOTAL
    cond = pd.Series(False, index=df2.index)
    if "source_csv" in df2.columns:
        cond = cond | (df2["source_csv"] == "MERGED")
    if "img" in df2.columns:
        cond = cond | (df2["img"] == "TOTAL")

    hit = df2[cond]
    if not hit.empty:
        return hit.iloc[-1]

    # 再找四個指標有值的列
    valid = df2.dropna(subset=METRIC_COLS, how="all")
    if not valid.empty:
        return valid.iloc[-1]

    return None

rows = []


for csv_path in sorted(ROOT.rglob("*.csv")):
    if csv_path.name not in TARGET_NAMES:
        continue

    rel_parts = csv_path.relative_to(ROOT).parts

    # 只吃當前 group，避免跑 FG 時把舊 BGC 一起整理進 summary
    if len(rel_parts) < 1 or rel_parts[0] != target_group:
        continue

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"[讀取失敗] {csv_path}: {e}")
        continue

    missing = [c for c in METRIC_COLS if c not in df.columns]
    if missing:
        print(f"[缺欄位] {csv_path} -> 缺少 {missing}")
        continue

    metric_row = pick_metric_row(df)
    if metric_row is None:
        print(f"[找不到整體指標] {csv_path}")
        continue



    group_type = rel_parts[0] if len(rel_parts) >= 1 else None   # FG / BGC
    map_name   = None
    run_name   = None
    scope      = "single"

    if len(rel_parts) >= 2:
        if rel_parts[1].startswith("merged_reports_all_"):
            map_name = "ALL"
            run_name = "merged"
            scope = "merged"
        else:
            map_name = rel_parts[1]

    if len(rel_parts) >= 3 and scope != "merged":
        run_name = rel_parts[2]

    if "image_view" in csv_path.name:
        report_kind = "image_view_black_le50"
    elif "vote_view" in csv_path.name:
        report_kind = "vote_view"
    else:
        report_kind = csv_path.stem

    rows.append({
        "group_type": group_type,
        "map_name": map_name,
        "run_name": run_name,
        "scope": scope,
        "report_kind": report_kind,
        "csv_name": csv_path.name,
        "relative_path": str(csv_path.relative_to(ROOT)),
        "overall_acc": metric_row["overall_acc"],
        "micro_f1": metric_row["micro_f1"],
        "macro_f1": metric_row["macro_f1"],
        "weighted_f1": metric_row["weighted_f1"],
    })

summary_df = pd.DataFrame(rows)
# ====== 欄位順序調整 ======
col_order = [
    "group_type",
    "map_name",
    "report_kind",
    "overall_acc",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
    "run_name",
    "scope",
    "csv_name",
    "relative_path",
]

# 只保留實際存在的欄位，避免報錯
col_order = [c for c in col_order if c in summary_df.columns]
summary_df = summary_df[col_order]
if summary_df.empty:
    print("沒有找到符合條件的報表。")
else:
    for c in METRIC_COLS:
        summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce").round(4)

    # ====== 自訂排序：FG 先、BGC 後；每組內 ALL 先、其他後 ======
    summary_df["_group_order"] = summary_df["group_type"].map({
        "FG": 0,
        "BGC": 1,
    }).fillna(99)
    
    summary_df["_all_order"] = summary_df["map_name"].astype(str).apply(
        lambda x: 0 if x == "ALL" else 1
    )
    
    summary_df["_map_name_str"] = summary_df["map_name"].astype(str)
    
    summary_df = summary_df.sort_values(
        by=[
            "_group_order",   # FG -> BGC
            "_all_order",     # ALL -> others
            "_map_name_str",  # 其他 map_name 再排
            "scope",
            "run_name",
            "report_kind",
            "csv_name",
        ],
        na_position="last"
    ).reset_index(drop=True)
    
    # 排完把輔助欄位刪掉
    summary_df = summary_df.drop(columns=["_group_order", "_all_order", "_map_name_str"])
    
    print(f"共找到 {len(summary_df)} 份報表")
    print(summary_df)

    out_csv = ROOT / "summary_metrics_4cols.csv"
    
    # =========================================================
    # Update summary_metrics_4cols.csv by group
    # ---------------------------------------------------------
    # Current run only recomputes target_group:
    #   legends -> FG
    #   others  -> BGC
    #
    # Therefore:
    #   - remove old rows of current target_group
    #   - keep rows of the other group
    #   - append current target_group rows
    # =========================================================
    if out_csv.exists():
        old_df = pd.read_csv(out_csv, encoding="utf-8-sig")
    
        if "group_type" in old_df.columns:
            old_df["group_type"] = old_df["group_type"].astype(str).str.strip()
            old_keep = old_df[old_df["group_type"] != target_group].copy()
        else:
            print(f"[WARN] Existing summary has no group_type column, will rebuild from current {target_group} only.")
            old_keep = pd.DataFrame(columns=summary_df.columns)
    
        summary_to_save = pd.concat([old_keep, summary_df], ignore_index=True, sort=False)
    else:
        summary_to_save = summary_df.copy()
    
    # 欄位順序統一
    for c in col_order:
        if c not in summary_to_save.columns:
            summary_to_save[c] = np.nan
    
    summary_to_save = summary_to_save[col_order]
    
    # 指標轉數字與 round
    for c in METRIC_COLS:
        if c in summary_to_save.columns:
            summary_to_save[c] = pd.to_numeric(summary_to_save[c], errors="coerce").round(4)
    
    # 重新排序：FG 先、BGC 後；ALL 先、其他後
    summary_to_save["_group_order"] = summary_to_save["group_type"].map({
        "FG": 0,
        "BGC": 1,
    }).fillna(99)
    
    summary_to_save["_all_order"] = summary_to_save["map_name"].astype(str).apply(
        lambda x: 0 if x == "ALL" else 1
    )
    
    summary_to_save["_map_name_str"] = summary_to_save["map_name"].astype(str)
    
    summary_to_save = summary_to_save.sort_values(
        by=[
            "_group_order",
            "_all_order",
            "_map_name_str",
            "scope",
            "run_name",
            "report_kind",
            "csv_name",
        ],
        na_position="last"
    ).reset_index(drop=True)
    
    summary_to_save = summary_to_save.drop(columns=["_group_order", "_all_order", "_map_name_str"])
    
    summary_to_save.to_csv(out_csv, index=False, encoding="utf-8-sig")
    
    print(f"\n已更新：{out_csv}")
    print(f"[INFO] Updated group: {target_group}")
    print(f"[INFO] Total rows in summary: {len(summary_to_save)}")



###################


# =========================================================
# Optional Part: GeoNet-vs-Ours comparison table
# ---------------------------------------------------------
# The previous sections already generated the current method
# summary report:
#   reports/{method}/summary_metrics_4cols.csv
#
# The following comparison table requires BOTH:
#   reports/GeoNet_Orig_augmentation/summary_metrics_4cols.csv
#   reports/gen_legend_ratio_augmentation/summary_metrics_4cols.csv
#
# Therefore, skip it by default unless --run_compare_table is given.
# =========================================================
if not config.args.run_compare_table:
    print("[DONE] Current classifier summary has been generated.")
    print("[SKIP] GeoNet-vs-Ours comparison table is disabled.")
    print("       Add --run_compare_table if both GeoNet and Ours summaries exist.")
    raise SystemExit(0)

# ====== 0) 你只要改這裡 ======
from pathlib import Path

CSV_CONFIG = [
    {
        "path": "reports/GeoNet_Orig_augmentation/summary_metrics_4cols.csv",
        "method": "GeoNet",
    },
    {
        "path": "reports/gen_legend_ratio_augmentation/summary_metrics_4cols.csv",
        "method": "Ours",
    },
]

OUT_DIR = Path("reports/compare_summary_tables")
dpi = 300
save_pdf = False   # True 會另外存 PDF

# ====== 1) 讀取資料 ======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["overall_acc", "macro_f1", "weighted_f1"]

METRIC_LABELS = {
    "overall_acc": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
}

REPORT_KIND_TO_SUB = {
    "image_view_black_le50": "Img",
    "vote_view": "Vote",
}

required_cols = ["group_type", "map_name", "report_kind"] + METRICS

dfs = []
for cfg in CSV_CONFIG:
    csv_path = Path(cfg["path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到檔案：{csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"{csv_path} 缺少必要欄位：{miss}")

    use = df[required_cols].copy()
    use["method"] = cfg["method"]

    use["group_type"] = use["group_type"].astype(str).str.strip()
    use["map_name"] = use["map_name"].astype(str).str.strip()
    use["report_kind"] = use["report_kind"].astype(str).str.strip()

    for c in METRICS:
        use[c] = pd.to_numeric(use[c], errors="coerce")

    dfs.append(use)

all_df = pd.concat(dfs, ignore_index=True)

# 若有重複 key，保留第一筆
dedup_key = ["method", "group_type", "map_name", "report_kind"]
dup_mask = all_df.duplicated(dedup_key, keep=False)
if dup_mask.any():
    print("[警告] 發現重複資料，將保留第一筆：")
    display(all_df.loc[dup_mask, dedup_key + METRICS].sort_values(dedup_key))
    all_df = all_df.drop_duplicates(subset=dedup_key, keep="first")

# ====== 2) 排序規則 ======
def map_sort_key(x):
    s = str(x).strip()
    if s == "ALL":
        return (0, 0, "")
    if s.isdigit():
        return (1, int(s), "")
    return (2, 999999, s)

method_order = [cfg["method"] for cfg in CSV_CONFIG]

# ====== 3) 建立給畫表用的 DataFrame ======
def build_plot_df(group_type: str):
    sub = all_df[all_df["group_type"] == group_type].copy()
    if sub.empty:
        raise ValueError(f"沒有 {group_type} 的資料")

    map_names = sorted(sub["map_name"].dropna().unique().tolist(), key=map_sort_key)

    rows = []
    for metric in METRICS:
        for method in method_order:
            row = {
                "Metric": METRIC_LABELS[metric],
                "Method": method,
            }
            sub_m = sub[sub["method"] == method]

            for mp in map_names:
                for rk in ["image_view_black_le50", "vote_view"]:
                    col_name = f"{mp} {REPORT_KIND_TO_SUB[rk]}"
                    hit = sub_m[
                        (sub_m["map_name"] == mp) &
                        (sub_m["report_kind"] == rk)
                    ]

                    if hit.empty:
                        row[col_name] = np.nan
                    else:
                        row[col_name] = hit.iloc[0][metric]

            rows.append(row)

    plot_df = pd.DataFrame(rows)

    ordered_cols = []
    for mp in map_names:
        ordered_cols.extend([f"{mp} Img", f"{mp} Vote"])

    plot_df = plot_df[["Metric", "Method"] + ordered_cols]
    return plot_df, map_names, ordered_cols

# ====== 4) 畫表函式（沿用你之前那版風格） ======
def draw_table(df, groups, ordered_cols, out_png, out_pdf=None):
    df = df.copy()

    # 顯示格式
    float_fmt = "{:.4f}"
    df_show = df.copy()
    for c in ordered_cols:
        df_show[c] = df_show[c].map(lambda x: "--" if pd.isna(x) else float_fmt.format(float(x)))

    # 雙層表頭
    header0 = ["Metric", "Method"] + [""] * len(ordered_cols)
    header1 = ["", ""] + sum([["Img", "Vote"] for _ in groups], [])

    # ====== 讓 Metric 同組只顯示一次（第二列清空） ======
    metric_vals = df_show["Metric"].tolist()
    for i in range(1, len(metric_vals)):
        if metric_vals[i] == metric_vals[i - 1]:
            df_show.at[df_show.index[i], "Metric"] = ""
    
    body = df_show.values.tolist()
    cell_text = [header0, header1] + body
    
    nrows = len(cell_text)
    ncols = len(header0)
    n_groups = len(groups)

    # 自適應大小
    if ncols <= 4:
        fig_w = 7.2
    elif ncols <= 6:
        fig_w = 9.0
    elif ncols <= 10:
        fig_w = 12.0
    else:
        fig_w = max(13.5, 1.15 * ncols)

    fig_h = max(2.8, 0.55 * nrows)

    if ncols <= 4:
        fs = 11
    elif ncols <= 8:
        fs = 10.5
    elif ncols <= 12:
        fs = 10
    else:
        fs = 9

    if ncols <= 4:
        left_w_metric = 0.22
        left_w_method = 0.18
    elif ncols <= 8:
        left_w_metric = 0.16
        left_w_method = 0.14
    else:
        left_w_metric = 0.12
        left_w_method = 0.10

    remain = 1.0 - left_w_metric - left_w_method
    num_w = remain / (ncols - 2)
    col_widths = [left_w_metric, left_w_method] + [num_w] * (ncols - 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        colWidths=col_widths
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fs)

    if ncols <= 4:
        tbl.scale(1.0, 1.45)
    else:
        tbl.scale(1.0, 1.35)

    # 全白底、取消內建格線
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_linewidth(0.0)
        cell.set_edgecolor("white")
        cell.PAD = 0.18

    # 表頭加粗
    for c in range(ncols):
        tbl[(0, c)].get_text().set_weight("bold")
        tbl[(1, c)].get_text().set_weight("bold")

    fig.canvas.draw()

    # 邊界座標
    left  = tbl[(0, 0)].get_x()
    right = tbl[(0, ncols - 1)].get_x() + tbl[(0, ncols - 1)].get_width()
    top   = tbl[(0, 0)].get_y() + tbl[(0, 0)].get_height()
    head_bottom = tbl[(1, 0)].get_y()
    bottom = tbl[(nrows - 1, 0)].get_y()

    # 三線表主線
    ax.hlines(top, left, right, colors="black", linewidth=1.6, transform=ax.transAxes, zorder=10)
    ax.hlines(head_bottom, left, right, colors="black", linewidth=1.2, transform=ax.transAxes, zorder=10)
    ax.hlines(bottom, left, right, colors="black", linewidth=1.6, transform=ax.transAxes, zorder=10)

    # 群組標題（跨兩欄置中）
    for gi, gname in enumerate(groups):
        c0 = 2 + gi * 2
        c1 = c0 + 1
        x0 = tbl[(0, c0)].get_x()
        x1 = tbl[(0, c1)].get_x() + tbl[(0, c1)].get_width()
        xc = (x0 + x1) / 2
        yc = tbl[(0, c0)].get_y() + tbl[(0, c0)].get_height() * 0.56

        ax.text(
            xc, yc, str(gname),
            ha="center", va="center",
            fontsize=fs, fontweight="bold",
            transform=ax.transAxes, zorder=20
        )

    # 垂直分隔線
    x_after_metric = tbl[(0, 0)].get_x() + tbl[(0, 0)].get_width()
    ax.vlines(x_after_metric, bottom, top, colors="black", linewidth=1.2,
              transform=ax.transAxes, zorder=15)

    x_after_method = tbl[(0, 1)].get_x() + tbl[(0, 1)].get_width()
    ax.vlines(x_after_method, bottom, top, colors="black", linewidth=1.2,
              transform=ax.transAxes, zorder=15)

    for gi in range(n_groups):
        c_vote = 2 + gi * 2 + 1
        x = tbl[(0, c_vote)].get_x() + tbl[(0, c_vote)].get_width()
        ax.vlines(x, bottom, top, colors="black", linewidth=0.8,
                  transform=ax.transAxes, zorder=15)

    # Metric 分組水平線
    metric_col = df_show["Metric"].replace("", np.nan).ffill().astype(str).tolist()
    for i in range(1, len(metric_col)):
        if metric_col[i] != metric_col[i - 1]:
            r = 2 + i
            y = tbl[(r, 0)].get_y() + tbl[(r, 0)].get_height()
            ax.hlines(y, left, right, colors="black", linewidth=0.9,
                      transform=ax.transAxes, zorder=12)

    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    if out_pdf is not None:
        plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08, facecolor="white")

    plt.show()

    print("Saved:", out_png)
    if out_pdf is not None:
        print("Saved:", out_pdf)

# ====== 5) 分別輸出 FG / BGC ======
for group_type in ["FG", "BGC"]:
    plot_df, groups, ordered_cols = build_plot_df(group_type)

    out_png = OUT_DIR / f"{group_type}_summary_table.png"
    out_pdf = OUT_DIR / f"{group_type}_summary_table.pdf" if save_pdf else None

    draw_table(
        df=plot_df,
        groups=groups,
        ordered_cols=ordered_cols,
        out_png=out_png,
        out_pdf=out_pdf
    )