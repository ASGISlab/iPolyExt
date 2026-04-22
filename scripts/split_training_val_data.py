# -*- coding: utf-8 -*-
from pathlib import Path
import re, math, hashlib, sys, yaml, json, shutil
import pandas as pd
try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
from src.augment_and_train_classifier.split_train_val import main as split_train_val_main
from src.augment_and_train_classifier.generate_train_data_manifest import main as generate_train_data_manifest_main

current_map = config.args.map
method = config.args.cls_method


cfg = Path("configs/augment_and_train_classifier/split_train_val.yaml")

overrides = {
    "src": str(Path(f"{config.args.cls_dir}/stage_2") /method/ current_map),
    "out_base": str(Path(f"{config.args.cls_dir}/stage_3") /method/ current_map/config.args.run_name),
    "default_map_name": current_map,
}

with open(cfg, "r", encoding="utf-8") as f:
    d = yaml.safe_load(f)
d.update(overrides)
split_train_val_main(d)



# ===== 0) 你的 fixed_train_data base =====

OUT_BASE = Path(f"{config.args.cls_dir}/stage_3")/method / current_map/config.args.run_name

SPLIT_DIR = OUT_BASE
print("✅ using:", SPLIT_DIR)

TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR   = SPLIT_DIR / "val"
assert TRAIN_DIR.exists() and VAL_DIR.exists(), "train/val 資料夾不存在"

# ===== 1) 檔名解析（bg/op/nz）=====
_re_bg = re.compile(r"__bg-([A-Za-z0-9]+)")
_re_op = re.compile(r"__op-(\d{3})")

def _strip_tail_index(s: str) -> str:
    # 去掉最後的 __123 或 _123
    return re.sub(r"(?:__|_)\d+$", "", s)
def parse_fields_from_filename(name: str):
    stem = Path(name).stem

    m_bg = _re_bg.search(stem)
    bg = m_bg.group(1) if m_bg else "UNK"

    m_op = _re_op.search(stem)
    op = m_op.group(1) if m_op else "UNK"

    # 預設
    nz_mode, nz_bin, nz_tag = "UNK", "unk", "NONE"
    nz_pct = None  # ✅ 新增：解析到的白遮罩百分比（int），解析不到就 None

    if "__nz-" in stem:
        nz_part = stem.split("__nz-", 1)[1]
        nz_tag = _strip_tail_index(nz_part)      # 舊：L25-24 / R50-51 / N00-00；新：L13 / R27 / N00
        nz_mode = nz_tag[0] if nz_tag else "UNK" # N/L/R

        if nz_mode == "N":
            nz_bin = "none"
            # 可能 N00 或 N00-00
            m = re.match(r"^N(\d{2})(?:-(\d{2}))?$", nz_tag)
            if m:
                # 舊版 N00-00 / 新版 N00 都算
                nz_pct = int(m.group(2) or m.group(1))

        elif nz_mode in ("L", "R"):
            # 舊版：L25-24 / R50-51
            m_old = re.match(r"^[LR](25|50)-(\d{2})$", nz_tag)
            if m_old:
                bin_code, pct = m_old.group(1), m_old.group(2)
                nz_bin = "le25" if bin_code == "25" else "ge25"
                nz_pct = int(pct)
            else:
                # 新版：L13 / R27（只有百分比）
                m_new = re.match(r"^[LR](\d{2})$", nz_tag)
                if m_new:
                    nz_bin = "single"
                    nz_pct = int(m_new.group(1))
                else:
                    nz_bin = "unk"

        else:
            nz_bin = "unk"

    else:
        nz_mode, nz_bin, nz_tag = "NO_NZ", "none", "NO_NZ"
        nz_pct = None

    return bg, op, nz_mode, nz_bin, nz_tag, nz_pct


def collect(split_root: Path, split_name: str):
    rows = []
    for p in split_root.rglob("*.png"):
        rel = p.relative_to(split_root).parts
        if not rel:
            continue
        label = rel[0]                    # train/<label>/...
        map_name = None
        if label not in ("BGC", "blank") and len(rel) >= 2:
            map_name = rel[1]            # train/<label>/<MAP>/...
        bg, op, nz_mode, nz_bin, nz_tag, nz_pct = parse_fields_from_filename(p.name)
        rows.append({
            "split": split_name,
            "label": label,
            "map": map_name,
            "name": p.name,
            "bg": bg,
            "op": op,
            "nz_mode": nz_mode,
            "nz_bin": nz_bin,
            "nz_tag": nz_tag,
            "nz_pct": nz_pct,
            "path": str(p),
        })
    return rows

rows = collect(TRAIN_DIR, "train") + collect(VAL_DIR, "val")
df = pd.DataFrame(rows)
print("✅ total png:", len(df), "| labels:", df["label"].nunique())

# ===== 2) (A) 先檢查每個 label 的 8:2 =====
VAL_RATIO = 0.2

cnt = df.groupby(["label","split"]).size().unstack(fill_value=0)
cnt["total"] = cnt.get("train",0) + cnt.get("val",0)
cnt["val_expected_round"] = (cnt["total"] * VAL_RATIO).round().astype(int)
cnt["val_diff"] = cnt.get("val",0) - cnt["val_expected_round"]
cnt["val_ratio"] = cnt.get("val",0) / cnt["total"].replace(0, 1)

print("\n=== [Check] 8:2 per label (val should be round(total*0.2)) ===")
display(cnt.sort_values(["val_diff","total"], ascending=[False, False]))

bad = cnt[ cnt["val_diff"] != 0 ]
print("\nlabels with val_diff != 0:", len(bad))
if len(bad):
    display(bad.sort_values("val_diff", ascending=False))

# ===== 3) (B) 針對某個 label 做「比例檢查」=====
FOCUS_LABEL = "Db"   # 你想看哪個類別就改這裡；不存在就會自動跳過

sub = df[df["label"] == FOCUS_LABEL].copy()
if len(sub) == 0:
    print(f"\n⚠️ 單類別檢查 找不到 label={FOCUS_LABEL}，已跳過 (B)(C) 比例檢查。")
    print("可用 labels：")
    # 顯示所有 label 與總數（讓你直接複製去改 FOCUS_LABEL）
    label_list = (
        df.groupby("label")
          .size()
          .sort_values(ascending=False)
    )
    display(label_list)
else:
    def show_counts(title, data, col, order=None):
        tab = data.groupby(["split", col]).size().unstack(fill_value=0)
        if order is not None:
            for k in order:
                if k not in tab.columns:
                    tab[k] = 0
            tab = tab[order]
        tab["total"] = tab.sum(axis=1)
        pct = (tab.div(tab["total"], axis=0) * 100).round(2)
        print(f"\n=== {title} ({FOCUS_LABEL}) ===")
        display(tab)
        print("percent (%):")
        display(pct)

    # (1) opacity：3:1:1:1:1
    op_order = ["100","075","050","025","000"]
    show_counts("Opacity (expect 3:1:1:1:1)", sub, "op", op_order)

    # (2) bg：2:1:1 只在 op!=000 內看（因為 op=000 是 bg-NONE）
    sub_nonzero_op = sub[sub["op"] != "000"].copy()
    bg_order = ["CS","C5K","C5K2"]  # 依你 BG_CODE；若你實際 code 不同就改
    show_counts("BG among op!=000 (expect 2:1:1 on {CS,C5K,C5K2})", sub_nonzero_op, "bg", bg_order)

    # (3) nz_mode：line/rect/none = 1:1:1  （你的 tag 是 L/R/N）
    nz_mode_order = ["L","R","N","NO_NZ","UNK"]
    show_counts("NZ mode (expect L:R:N = 1:1:1)", sub, "nz_mode", nz_mode_order)

    # (4) nz_bin：舊版才有 le25/ge25 = 1:3；新版只有 single（不做 1:3）
    for mode in ["L", "R"]:
        s2 = sub[sub["nz_mode"] == mode].copy()
        if len(s2) == 0:
            print(f"\n=== NZ bin for mode={mode}: (no samples) ===")
            continue

        bins = set(s2["nz_bin"].astype(str).tolist())
        has_legacy = ("le25" in bins) or ("ge25" in bins)

        if has_legacy:
            show_counts(
                f"NZ bin within mode={mode} (legacy expect le25:ge25 = 1:3)",
                s2, "nz_bin", ["le25", "ge25", "single", "unk", "none"]
            )
        else:
            show_counts(
                f"NZ bin within mode={mode} (new: usually 'single'; no le25/ge25 check)",
                s2, "nz_bin", ["single", "unk", "none"]
            )

            if "nz_pct" in s2.columns:
                s2_num = s2.dropna(subset=["nz_pct"])
                if len(s2_num):
                    desc = s2_num.groupby("split")["nz_pct"].agg(["count", "mean", "std", "min", "max"]).round(3)
                    print(f"\n[nz_pct stats] mode={mode} (percentage extracted from tag)")
                    display(desc)

    # ===== 4) (C) 進階：檢查「每個 recipe 在 train/val 是否約 8:2」=====
    sub["recipe"] = list(zip(sub["bg"], sub["op"], sub["nz_mode"], sub["nz_bin"]))
    rt = sub.groupby(["recipe","split"]).size().unstack(fill_value=0)
    rt["total"] = rt.sum(axis=1)
    rt["val_ratio"] = rt.get("val",0) / rt["total"].replace(0,1)
    print("\n=== [Check] recipe-level val_ratio (should be around 0.2; small buckets may deviate due to rounding) ===")
    display(rt.sort_values("val_ratio").head(10))
    display(rt.sort_values("val_ratio", ascending=False).head(10))

if method=="gen_legend_ratio_augmentation":
    cfg    = Path("configs/augment_and_train_classifier/generate_train_data_manifest.yaml")
    base_dir = Path(f"{config.args.cls_dir}/stage_3")/method / current_map/config.args.run_name

    # 先用最新那個當候選
    cand = base_dir
    print("✅ base_dir:", base_dir)
    print("✅ picked:", cand)

    # 1) 先找 cand/meta.json
    meta_path = cand / "meta.json"

    # 2) 如果沒有，往下找任何 meta.json，並把 root_dir 指到 meta.json 所在資料夾
    if not meta_path.exists():
        hits = list(cand.rglob("meta.json"))
        if hits:
            meta_path = hits[0]
            cand = meta_path.parent
            print("⚠️ meta.json 不在 fixed_train_data_* 根目錄，改用：", cand)
        else:
            raise FileNotFoundError(f"在 {cand} 底下找不到任何 meta.json")

    print("✅ meta_path:", meta_path)
    print("✅ out_path :", cand / "train_data.json")

    # （可選）快速檢查 meta.json 是否真的有 classes
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print("✅ meta keys:", list(meta.keys())[:20])
    print("✅ has classes:", "classes" in meta, "type:", type(meta.get("classes")).__name__)

    overrides = {
        "root_dir": str(cand),
    }

    with open(cfg, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    d.update(overrides)

    print(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    generate_train_data_manifest_main(d)


else:#geonet
    from datetime import datetime
    from zoneinfo import ZoneInfo

    # =========================
    # 0) 你要填的參數
    # =========================
    
    ROOT = Path(f"{config.args.cls_dir}/stage_3")/method / current_map/config.args.run_name

    AUTO_PICK_LATEST = True
    DATASET_DIR = None  # 例如：ROOT/"fixed_train_data_20250923_061359"

    # 特殊類別：key/core 都原樣，不做任何 suffix / padding
    SPECIAL_CLASSES = {"blank", "bg_color","BGC"}
    # 額外排除：這些資料夾不要當成類別
    EXCLUDE_CLASS_DIRS = {".ipynb_checkpoints"}
    # 參考 JSON：用來推斷哪些 core 需要 core_13 (例如 Ms_13 / Mst_13 / Pi_13)
    REF_JSON_PATH = Path(f"{config.args.cls_dir}/stage_3")/method / current_map/config.args.run_name/"train_data.json"
    #Path("data/train_data/stage_3/gen_legend_ratio/13/fixed_train_data_20260129_140459)

    # =========================
    # 1) 工具函式
    # =========================
    def now_taipei_str():
        dt = datetime.now(ZoneInfo("Asia/Taipei"))
        return dt.strftime("%Y-%m-%d %H:%M:%S"), dt.strftime("%Y%m%d_%H%M%S")

    def infer_target_tag_from_root(root: Path) -> str | None:
        # 例如 "color correction_13" -> "13"
        m = re.search(r"_(\d+)\s*$", root.name)
        return m.group(1) if m else None

    def extract_core(name: str) -> str:
        # 允許像 Mag-Mk_17_13 這種：取最後一段 "-" 後，再取 "_" 前
        s = name.split("-")[-1]
        return s.split("_")[0]

    def pad_numeric_tokens_if_multi(key: str) -> str:
        # 只有當 key 內「數字 token >= 2」才把所有數字 token zfill(3)
        toks = key.split("_")
        num_cnt = sum(1 for t in toks if t.isdigit())
        if num_cnt >= 2:
            toks = [t.zfill(3) if t.isdigit() else t for t in toks]
        return "_".join(toks)

    def load_ref_suffix_policy(ref_json: Path, target_tag: str | None):
        """
        從參考 JSON 推斷：哪些 core 會用 target_tag 做 suffix（例如 Ms_13）
        回傳 set[str]：cores_force_target_suffix
        """
        cores_force = set()
        if not (ref_json and ref_json.exists()):
            return cores_force

        with ref_json.open("r", encoding="utf-8") as f:
            ref = json.load(f)

        classes = ref.get("classes", {})
        for key, info in classes.items():
            core = info.get("core", extract_core(key))
            if target_tag and key == f"{core}_{target_tag}":
                cores_force.add(core)
        return cores_force

    def choose_dataset_dir(root: Path) -> Path:
        if DATASET_DIR is not None:
            return Path(DATASET_DIR)

        cands = sorted([d for d in root.iterdir() if d.is_dir()],
                    key=lambda p: p.name)
        if not cands:
            raise FileNotFoundError(f"ROOT 底下找不到 fixed_train_data_*：{root}")

        return cands[-1] if AUTO_PICK_LATEST else cands[0]

    def make_class_key(cname: str, cores_force_target_suffix: set, target_tag: str | None) -> str:
        """
        產生 classes 的 key（可能和 core 不同），符合你貼的例子：Qy_1/core=Qy
        規則：
        - SPECIAL_CLASSES: key=原樣，不做任何處理
        - 若 cname 已經帶 suffix：直接用（必要時多數字才補 3 位）
        - 若 core 在「參考 JSON」中屬於 target suffix 類（如 Ms/Mst/Pi），且 target_tag 存在：
            key = core_targetTag（例如 Ms_13）
        - 否則：key=原樣
        - 最後：若 key 內有 >=2 個數字 token，全部數字 token zfill(3)（Mm_18_13_17 -> Mm_018_013_017）
        """
        if cname in SPECIAL_CLASSES:
            return cname

        core = extract_core(cname)

        # 已有 suffix
        if "_" in cname:
            return pad_numeric_tokens_if_multi(cname)

        # 參考策略：這些 core 會固定用 target_tag
        if target_tag and core in cores_force_target_suffix:
            return pad_numeric_tokens_if_multi(f"{core}_{target_tag}")

        # 原樣
        return pad_numeric_tokens_if_multi(cname)

    # =========================
    # 2) 主流程：掃資料夾 -> 產生 JSON -> 寫檔
    # =========================
    assert ROOT.exists() and ROOT.is_dir(), f"ROOT 不存在或不是資料夾：{ROOT}"

    created_at_str, ts = now_taipei_str()
    target_tag = infer_target_tag_from_root(ROOT)
    cores_force_target_suffix = load_ref_suffix_policy(REF_JSON_PATH, target_tag)

    ds = choose_dataset_dir(ROOT)
    train_dir = ds / "train"
    val_dir = ds / "val"
    assert train_dir.exists() and val_dir.exists(), f"找不到 train/val：{ds}"

    # 抓 train/val 第一層類別資料夾（取 union）
    train_classes = {p.name for p in train_dir.iterdir() if p.is_dir()}
    val_classes   = {p.name for p in val_dir.iterdir() if p.is_dir()}

    all_classes = sorted(((train_classes | val_classes) - EXCLUDE_CLASS_DIRS), key=lambda s: s.lower())


    classes_dict = {}
    for cname in all_classes:
        key = make_class_key(cname, cores_force_target_suffix, target_tag)
        core = key if key in SPECIAL_CLASSES else extract_core(key)

        if key in classes_dict:
            raise RuntimeError(f"[重複 key] {key} 來源類別可能衝突（例如同時出現 {cname} 與其他同 key）")

        classes_dict[key] = {
            "core": core,
            "components": [core],
            "attrs": {"pure_color": False},
            "extra": {}
        }

    unique_cores = sorted({v["core"] for v in classes_dict.values()}, key=lambda s: s.lower())

    out = {
        "version": 2,
        "created_at": created_at_str,
        "meta_ref": "meta.json",
        "attributes": {
            "pure_color": {
                "description": "是否屬於純色圖（無紋理）",
                "type": "boolean",
                "enabled": True,
                "default": False,
                "positives": []
            }
        },
        "classes": classes_dict,
        "derived": {
            "unique_cores": unique_cores
        }
    }

    out_path = ds / "train_data.json"
    if out_path.exists():
        bak = ds / f"train_data.json.bak_{ts}"
        shutil.copy2(out_path, bak)
        print(f"[backup] 已備份舊檔 -> {bak}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已寫入：{out_path}")
    print(f"[created_at] {created_at_str} (Asia/Taipei)")
    print(f"[dataset_dir] {ds}")
    print(f"[classes] {len(classes_dict)}")
    print(f"[derived.unique_cores] {len(unique_cores)}")

    print("\n[preview keys]")
    for k in list(classes_dict.keys())[:60]:
        print(" -", k)
    if len(classes_dict) > 60:
        print(f"...（省略 {len(classes_dict)-60} 個）")
