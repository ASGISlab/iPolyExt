import re
import ast
import json
import random
from pathlib import Path
import sys

import tornado
import cv2
import numpy as np
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src import config

import matplotlib
matplotlib.use("webagg")  # 保留你同事這版習慣
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle


# =========================
# 基本參數（外層格式照你同事）
# =========================
current_map = config.args.map
method = config.args.cls_method

valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

IN_DIR = f"{config.args.map_dir}/{current_map}"

# ROI 路徑格式照你同事
ROI_PATH = Path(f"{config.args.map_dir}/{current_map}.txt")

# 若手動框選後是否自動存檔
AUTO_SAVE_ROI = True

# ROI 最小尺寸防呆（你之後要改可直接改這裡）
MIN_ROI_W = 1
MIN_ROI_H = 1

# BGC 輸出格式照你同事
OUT_DIR = f"{config.args.cls_dir}/stage_2/{method}/{current_map}"
CLASS_NAME = "BGC"
TARGET_HW = (68, 110)   # (H, W)

if method == "gen_legend_ratio_augmentation":
    TOTAL_OUT = 280
else:
    TOTAL_OUT = 300

WRITE_JSON = True
RANDOM_SEED = 1234

rng = random.Random(RANDOM_SEED)


# =========================
# 工具函式
# =========================
def resolve_one_image(p: Path, valid_ext: set):
    if p.exists() and p.is_file() and p.suffix.lower() in valid_ext:
        return str(p)

    if p.exists() and p.is_dir():
        hits = sorted([x for x in p.glob("*") if x.suffix.lower() in valid_ext])
        if not hits:
            raise FileNotFoundError(f"找不到影像：{p}")
        return str(hits[0])

    parent = p.parent
    name = p.name
    hits = [x for x in parent.glob(name + ".*") if x.suffix.lower() in valid_ext]
    if not hits:
        raise FileNotFoundError(f"找不到影像：{p}（也找不到 {parent}/{name}.*）")

    ext_priority = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]
    hits = sorted(
        hits,
        key=lambda x: ext_priority.index(x.suffix.lower()) if x.suffix.lower() in ext_priority else 999
    )
    return str(hits[0])


def normalize_roi(roi_xyxy):
    x1, y1, x2, y2 = map(int, roi_xyxy)
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return (x1, y1, x2, y2)


def clamp_roi(roi_xyxy, w, h):
    x1, y1, x2, y2 = roi_xyxy
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return (x1, y1, x2, y2)


def dedupe_rois(rois):
    out = []
    seen = set()
    for roi in rois:
        roi = tuple(map(int, roi))
        if roi not in seen:
            out.append(roi)
            seen.add(roi)
    return out


def normalize_one_roi(obj):
    if isinstance(obj, dict):
        if all(k in obj for k in ["x1", "y1", "x2", "y2"]):
            return normalize_roi([obj["x1"], obj["y1"], obj["x2"], obj["y2"]])
        raise ValueError(f"ROI dict 格式錯誤：{obj}")

    if isinstance(obj, (list, tuple)) and len(obj) == 4:
        return normalize_roi(obj)

    raise ValueError(f"無法解析 ROI：{obj}")


def parse_roi_file(fp: Path):
    """
    支援：
    1) 舊格式單框
       ROI_XYXY = (x1, y1, x2, y2)
       x1,y1,x2,y2
       [x1, y1, x2, y2]
       {"x1":..,"y1":..,"x2":..,"y2":..}

    2) 新格式多框
       {"rois": [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]}
       [[x1,y1,x2,y2], [x1,y1,x2,y2], ...]
       ROI_XYXY_LIST = [(...), (...)]
    """
    if not fp.exists():
        return []

    text = fp.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 先試 JSON
    try:
        obj = json.loads(text)

        # 單框 list
        if isinstance(obj, list) and len(obj) == 4 and all(isinstance(v, (int, float)) for v in obj):
            return [normalize_roi(obj)]

        # 多框 list
        if isinstance(obj, list) and len(obj) > 0:
            rois = []
            for v in obj:
                rois.append(normalize_one_roi(v))
            return rois

        # dict
        if isinstance(obj, dict):
            if all(k in obj for k in ["x1", "y1", "x2", "y2"]):
                return [normalize_roi([obj["x1"], obj["y1"], obj["x2"], obj["y2"]])]

            if "rois" in obj and isinstance(obj["rois"], list):
                rois = []
                for v in obj["rois"]:
                    rois.append(normalize_one_roi(v))
                return rois
    except Exception:
        pass

    # ROI_XYXY_LIST = [...]
    m = re.search(r"ROI_XYXY_LIST\s*=\s*(\[.*\]|\(.*\))", text, flags=re.S)
    if m:
        try:
            obj = ast.literal_eval(m.group(1))
            rois = []
            for v in obj:
                rois.append(normalize_one_roi(v))
            return rois
        except Exception:
            pass

    # ROI_XYXY = (...)
    m = re.search(r"ROI_XYXY\s*=\s*(\([^\)]*\)|\[[^\]]*\])", text, flags=re.S)
    if m:
        try:
            obj = ast.literal_eval(m.group(1))
            return [normalize_one_roi(obj)]
        except Exception:
            pass

    # 純數字保底
    nums = re.findall(r"-?\d+", text)
    if len(nums) == 4:
        return [normalize_roi(list(map(int, nums[:4])))]

    if len(nums) >= 8 and len(nums) % 4 == 0:
        rois = []
        for i in range(0, len(nums), 4):
            rois.append(normalize_roi(list(map(int, nums[i:i + 4]))))
        return rois

    raise ValueError(f"ROI 檔格式無法解析：{fp}")


def save_roi_file(fp: Path, roi_list):
    fp.parent.mkdir(parents=True, exist_ok=True)
    roi_list = [list(map(int, r)) for r in roi_list]
    payload = {"rois": roi_list}
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def roi_is_big_enough(roi_xyxy, img_shape, target_hw):
    Ht, Wt = target_hw
    x1, y1, x2, y2 = clamp_roi(roi_xyxy, img_shape[1], img_shape[0])
    rw = x2 - x1 + 1
    rh = y2 - y1 + 1
    return (rw >= Wt) and (rh >= Ht)


def random_crop_in_roi(img_bgr, roi_xyxy, target_hw):
    """
    你的新版邏輯：
    - 大 ROI：直接 random crop
    - 小 ROI：把 ROI 內容重複貼滿，再隨機裁出 TARGET_HW
    """
    Ht, Wt = target_hw
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_roi(roi_xyxy, w, h)

    roi_img = img_bgr[y1:y2 + 1, x1:x2 + 1].copy()
    rh, rw = roi_img.shape[:2]

    if rw <= 0 or rh <= 0:
        return None

    # ROI 足夠大：直接 crop
    if rw >= Wt and rh >= Ht:
        x0 = rng.randint(x1, x2 - Wt + 1)
        y0 = rng.randint(y1, y2 - Ht + 1)
        return img_bgr[y0:y0 + Ht, x0:x0 + Wt].copy()

    # ROI 太小：重複平鋪後再裁
    rep_y = int(np.ceil(Ht / rh)) + 1
    rep_x = int(np.ceil(Wt / rw)) + 1
    tiled = np.tile(roi_img, (rep_y, rep_x, 1))

    max_sy = max(0, tiled.shape[0] - Ht)
    max_sx = max(0, tiled.shape[1] - Wt)

    sy = rng.randint(0, max_sy)
    sx = rng.randint(0, max_sx)

    return tiled[sy:sy + Ht, sx:sx + Wt].copy()


# =========================
# 讀圖 + 讀 ROI
# =========================
p = Path(IN_DIR)
IMG_PATH = resolve_one_image(p, valid_ext)
img = np.array(Image.open(IMG_PATH).convert("RGB"))

H, W = img.shape[:2]

ROI_XYXY_LIST = parse_roi_file(ROI_PATH)
ROI_XYXY_LIST = [clamp_roi(r, W, H) for r in ROI_XYXY_LIST]
ROI_XYXY_LIST = dedupe_rois(ROI_XYXY_LIST)
INITIAL_ROI_SET = set(ROI_XYXY_LIST)


# =========================
# ROI 選框 UI（你的新版多 ROI）
# =========================
plt.close("all")
fig, ax = plt.subplots(figsize=(10, 7))

ZOOM_STEP = 0.8
ZOOM_OUT_STEP = 1.25
MIN_VIEW_W = 80
MIN_VIEW_H = 80


def clamp_view(x0, x1, y0, y1, w, h):
    view_w = x1 - x0
    view_h = y1 - y0

    if view_w > w:
        x0, x1 = 0, w
    else:
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if x1 > w:
            x0 -= (x1 - w)
            x1 = w

    if view_h > h:
        y0, y1 = 0, h
    else:
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if y1 > h:
            y0 -= (y1 - h)
            y1 = h

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    return x0, x1, y0, y1


def zoom_view(scale):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x0, x1 = min(xlim), max(xlim)
    y0, y1 = min(ylim), max(ylim)

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    new_w = (x1 - x0) * scale
    new_h = (y1 - y0) * scale

    new_w = max(MIN_VIEW_W, min(W, new_w))
    new_h = max(MIN_VIEW_H, min(H, new_h))

    nx0 = cx - new_w / 2
    nx1 = cx + new_w / 2
    ny0 = cy - new_h / 2
    ny1 = cy + new_h / 2

    nx0, nx1, ny0, ny1 = clamp_view(nx0, nx1, ny0, ny1, W, H)

    ax.set_xlim(nx0, nx1)

    if ylim[0] > ylim[1]:
        ax.set_ylim(ny1, ny0)
    else:
        ax.set_ylim(ny0, ny1)

    fig.canvas.draw_idle()


def reset_view():
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.canvas.draw_idle()


ax.imshow(img)
ax.set_axis_off()

status = ax.text(
    0.01, 0.99,
    "ROI count = 0",
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
    color="white"
)

roi_patches = []
rs = None


def draw_roi(ax, roi_xyxy, edgecolor="red", linewidth=2):
    x1, y1, x2, y2 = roi_xyxy
    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth
    )
    ax.add_patch(rect)
    return rect


def redraw_all_rois():
    global roi_patches

    for patch in roi_patches:
        try:
            patch.remove()
        except Exception:
            pass
    roi_patches = []

    for roi in ROI_XYXY_LIST:
        color = "lime" if roi in INITIAL_ROI_SET else "red"
        patch = draw_roi(ax, roi, edgecolor=color, linewidth=2)
        roi_patches.append(patch)

    status.set_text(
        f"ROI count = {len(ROI_XYXY_LIST)} | "
        f"min size = ({MIN_ROI_W}, {MIN_ROI_H}) | "
        f"+/= zoom in | - zoom out | 0 reset | z=undo | c=clear | s=save"
    )
    fig.canvas.draw_idle()


def do_save():
    save_roi_file(ROI_PATH, ROI_XYXY_LIST)
    print(f"已儲存 {len(ROI_XYXY_LIST)} 個 ROI：{ROI_PATH}")


def onselect(eclick, erelease):
    global ROI_XYXY_LIST

    if eclick.xdata is None or erelease.xdata is None:
        return

    x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
    x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))

    roi = clamp_roi((x1, y1, x2, y2), W, H)

    if roi[0] == roi[2] or roi[1] == roi[3]:
        print("不能新增 ROI：框太小或零面積")
        return

    roi_w = roi[2] - roi[0]
    roi_h = roi[3] - roi[1]

    if roi_w < MIN_ROI_W or roi_h < MIN_ROI_H:
        print(
            f"不能新增 ROI：{roi}，"
            f"目前寬高 = ({roi_w}, {roi_h})，"
            f"最少需要 寬 >= {MIN_ROI_W}、高 >= {MIN_ROI_H}"
        )
        return

    ROI_XYXY_LIST.append(roi)
    ROI_XYXY_LIST = dedupe_rois(ROI_XYXY_LIST)

    redraw_all_rois()
    print(f"新增 ROI：{roi} | 寬高 = ({roi_w}, {roi_h})")

    if AUTO_SAVE_ROI:
        do_save()


def on_key(event):
    global ROI_XYXY_LIST

    if event.key in ["+", "="]:
        zoom_view(ZOOM_STEP)

    elif event.key == "-":
        zoom_view(ZOOM_OUT_STEP)

    elif event.key == "0":
        reset_view()

    elif event.key == "z":
        if len(ROI_XYXY_LIST) > 0:
            removed = ROI_XYXY_LIST.pop()
            redraw_all_rois()
            print(f"已刪除最後一個 ROI：{removed}")
            if AUTO_SAVE_ROI:
                do_save()

    elif event.key == "c":
        ROI_XYXY_LIST = []
        redraw_all_rois()
        print("已清空全部 ROI")
        if AUTO_SAVE_ROI:
            do_save()

    elif event.key == "s":
        do_save()

    if event.key == "escape":
        plt.close(event.canvas.figure)
        event.canvas.stop_event_loop()
        tornado.ioloop.IOLoop.current().stop()

if len(ROI_XYXY_LIST) > 0:
    print(f"已讀取 {len(ROI_XYXY_LIST)} 個 ROI：{ROI_PATH}")
    for i, roi in enumerate(ROI_XYXY_LIST):
        print(f"[{i}] {roi}")
else:
    print("找不到 ROI 檔，請手動框選。")

redraw_all_rois()

ax.set_title(
    f"Drag to add ROI | min size = ({MIN_ROI_W}, {MIN_ROI_H}) | "
    f"+/= zoom in | - zoom out | 0 reset | z=undo | c=clear | s=save | Esc quit"
)

rs = RectangleSelector(
    ax,
    onselect,
    useblit=False,
    button=[1],
    interactive=False
)

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()


# =========================
# 多 ROI 平衡生成 BGC
# =========================
ROI_XYXY_LIST = dedupe_rois(ROI_XYXY_LIST)
'''
n_roi = len(ROI_XYXY_LIST)
if n_roi <= 0:
    raise ValueError("ROI 數量不可為 0，請先框選至少一個 ROI。")
'''
n_roi = len(ROI_XYXY_LIST)
if n_roi <= 0:
    print("[SKIP] ROI 數量為 0，略過 BGC / others 產生。")
    sys.exit(0)

out_class_dir = Path(OUT_DIR) / CLASS_NAME / CLASS_NAME
out_class_dir.mkdir(parents=True, exist_ok=True)

pil = Image.open(IMG_PATH).convert("RGB")
img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

ROI_XYXY_LIST = [clamp_roi(r, img_bgr.shape[1], img_bgr.shape[0]) for r in ROI_XYXY_LIST]

# 平均分配 TOTAL_OUT
# 例：
# 280, 3 ROI -> [94, 93, 93]
# 280, 4 ROI -> [70, 70, 70, 70]
base_count = TOTAL_OUT // n_roi
remainder = TOTAL_OUT % n_roi
PER_ROI_COUNTS = [
    base_count + (1 if i < remainder else 0)
    for i in range(n_roi)
]

stem = Path(IMG_PATH).stem
made = 0
max_count = max(PER_ROI_COUNTS)

for round_idx in range(max_count):
    for roi_idx, roi_xyxy in enumerate(ROI_XYXY_LIST):
        if round_idx >= PER_ROI_COUNTS[roi_idx]:
            continue

        crop = random_crop_in_roi(img_bgr, roi_xyxy, TARGET_HW)
        if crop is None:
            raise RuntimeError(f"ROI[{roi_idx}] 在實際裁切時失敗：{roi_xyxy}")

        out_name = f"{stem}_{made:03d}.png"
        out_path = out_class_dir / out_name

        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)

        if WRITE_JSON:
            meta = {
                "class": CLASS_NAME,
                "source_path": IMG_PATH,
                "target_hw": {"H": TARGET_HW[0], "W": TARGET_HW[1]},
                "roi_mode": "multi_roi",
                "roi_xyxy_original": [int(v) for v in roi_xyxy],
                "roi_index": int(roi_idx),
                "roi_count_total": int(n_roi),
                "per_roi_out": int(PER_ROI_COUNTS[roi_idx]),
                "per_roi_counts_all": [int(v) for v in PER_ROI_COUNTS],
                "total_out": int(TOTAL_OUT),
                "sample_index_global": int(made),
                "sample_index_within_roi": int(round_idx),
                "note": "random crop in ROI; near-even balanced across multiple ROIs"
            }
            with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        made += 1

print(f"✅ 完成：{made}/{TOTAL_OUT} 張")
print(f"✅ ROI 數量：{n_roi}")
print(f"✅ 各 ROI 產出數量：{PER_ROI_COUNTS}")
print("📁 輸出：", out_class_dir)
