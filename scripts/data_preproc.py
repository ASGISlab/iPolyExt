# ====== 0) 你只要改這裡 ======
from pathlib import Path
from cmaas_utils.types import AreaBoundary, CMAAS_Map, Layout, Legend, GeoReference, MapUnit, MapUnitType, Provenance
from cmaas_utils.io import loadLayoutJson, loadLegendJson
from cmaas_utils.utilities import  mask_and_crop
import sys
import os
import gc
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
IN_DIR  = Path(config.args.map_dir)     # 你的輸入資料夾
OUT_DIR = Path(config.args.map_out_dir)    # 你的輸出資料夾

# global-consistent 參數（同一組套用到所有圖）
ALPHA = 1.3     # 對比(>1 增強對比；=1 不變)
BETA  = -58.4     # 亮度(可為負；+變亮 -變暗)
GAMMA = 1.0     # gamma(>1 變暗、<1 變亮；=1 不變)

# ===== paper_tint 固定參數（不提供 CLI 修改） =====
PAPER_BGR = (225, 235, 245)
PAPER_STRENGTH = 1.0
PAPER_HIGHLIGHT_GAMMA = 1.8
PAPER_TILE_SIZE = 1024

GLOBAL_PRESERVE_PURE_WHITE = False
PAPER_PRESERVE_PURE_WHITE = False

import matplotlib
matplotlib.use("webagg")  
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import tornado

JSON_FORMAT="""
[
  {{
    "name": "map",
    "bounds": {0},
    "confidence": 1.0,
    "ocr_text": null,
    "color_estimation": null,
    "model": {{
      "model": "manual",
      "field": "layout",
      "id": null
    }}
  }}
]
"""

RECURSIVE = True            # True: 會遞迴處理子資料夾並保留相對路徑結構
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# ====== 1) 實作 ======

IN_DIR  = IN_DIR.resolve()
OUT_DIR = OUT_DIR.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert IN_DIR.exists(), f"IN_DIR 不存在：{IN_DIR}"
assert IN_DIR.is_dir(), f"IN_DIR 不是資料夾：{IN_DIR}"

#https://github.com/DARPA-CRITICALMAAS/uiuc-pipeline
def boundingBox(array):
    array = np.array(array).astype(int)
    min_xy = [min(array, key=lambda x: (x[0]))[0], min(array, key=lambda x: (x[1]))[1]]
    max_xy = [max(array, key=lambda x: (x[0]))[0], max(array, key=lambda x: (x[1]))[1]]
    return [min_xy, max_xy]



def build_lut(alpha: float, gamma: float, beta: float = 0.0) -> np.ndarray:
    """
    global-consistent LUT:
    1) 以 pivot 為中心做對比縮放（alpha）
    2) 再做 gamma
    beta 是額外亮度偏移（想保持整體不變就設 0）
    """
    x = np.arange(256, dtype=np.float32)

    # ✅ pivot-contrast（中灰固定在 pivot 附近，不會被整體抬亮）
    x = alpha * (x) + beta
    x = np.clip(x, 0, 255)

    if gamma != 1.0:
        x = ((x / 255.0) ** gamma) * 255.0

    lut = np.clip(np.round(x), 0, 255).astype(np.uint8)
    return lut


LUT = build_lut(alpha=ALPHA, gamma=GAMMA, beta=BETA)
def apply_global_consistent(img: np.ndarray) -> np.ndarray:
    """
    img: uint8, shape (H,W), (H,W,3) or (H,W,4)
    回傳：同 shape 的 uint8
    """
    if img.ndim == 2:  # grayscale
        out = cv2.LUT(img, LUT)
        if GLOBAL_PRESERVE_PURE_WHITE:
            out[img == 255] = 255
        return out

    if img.shape[2] == 4:  # BGRA
        bgr = img[:, :, :3]
        a   = img[:, :, 3]
        if GLOBAL_PRESERVE_PURE_WHITE:
            white_mask = np.all(bgr == 255, axis=2)
        out_bgr = cv2.LUT(bgr, LUT)
        if GLOBAL_PRESERVE_PURE_WHITE:
            out_bgr[white_mask] = 255
        return np.dstack([out_bgr, a])

    # BGR
    bgr = img
    if GLOBAL_PRESERVE_PURE_WHITE:
        white_mask = np.all(bgr == 255, axis=2)
    out = cv2.LUT(bgr, LUT)
    if GLOBAL_PRESERVE_PURE_WHITE:
        out[white_mask] = 255
    return out
def apply_paper_tint_tilewise(
    img_bgr: np.ndarray,
    paper_bgr=(225, 235, 245),
    strength=1.0,
    highlight_gamma=1.8,
    tile_size=1024,
    preserve_pure_white=False,
) -> np.ndarray:
    """
    分塊版紙色化：
    - 白色會往 paper_bgr 偏
    - 越亮影響越大
    - 暗部影響小
    - preserve_pure_white=True 時，原本純白保持不變
    """
    h, w = img_bgr.shape[:2]
    out = np.empty_like(img_bgr)

    scale = np.array(paper_bgr, dtype=np.float32) / 255.0

    for y in range(0, h, tile_size):
        y2 = min(y + tile_size, h)
        for x in range(0, w, tile_size):
            x2 = min(x + tile_size, w)

            tile = img_bgr[y:y2, x:x2].astype(np.float32)

            if preserve_pure_white:
                white_mask = np.all(tile == 255, axis=2)

            tinted = tile * scale[None, None, :]

            luma = (
                0.114 * tile[:, :, 0] +
                0.587 * tile[:, :, 1] +
                0.299 * tile[:, :, 2]
            ) / 255.0

            w_map = (np.clip(luma, 0.0, 1.0) ** highlight_gamma) * strength
            w_map = w_map[:, :, None]

            tile_out = tile * (1.0 - w_map) + tinted * w_map
            tile_out = np.clip(tile_out, 0, 255).astype(np.uint8)

            if preserve_pure_white:
                tile_out[white_mask] = 255

            out[y:y2, x:x2] = tile_out

            del tile, tinted, luma, w_map, tile_out

    gc.collect()
    return out

def get_effective_preproc_mode() -> str:
    mode = config.args.preproc_mode
    return mode

def apply_preprocess_by_mode(img: np.ndarray) -> np.ndarray:
    """
    統一前處理入口：
    - none
    - global_consistent
    - paper_tint
    """
    mode = config.args.preproc_mode
    if mode == "none":
        return img

    if mode == "global_consistent":
        return apply_global_consistent(img)

    if mode == "paper_tint":
        if img.ndim == 2:
            raise ValueError("paper_tint 目前只支援彩色圖，收到 grayscale 圖片")

        if img.shape[2] == 4:
            bgr = img[:, :, :3]
            a = img[:, :, 3]
            out_bgr = apply_paper_tint_tilewise(
                img_bgr=bgr,
                paper_bgr=PAPER_BGR,
                strength=PAPER_STRENGTH,
                highlight_gamma=PAPER_HIGHLIGHT_GAMMA,
                tile_size=PAPER_TILE_SIZE,
                preserve_pure_white=PAPER_PRESERVE_PURE_WHITE,
            )
            return np.dstack([out_bgr, a])

        return apply_paper_tint_tilewise(
            img_bgr=img,
            paper_bgr=PAPER_BGR,
            strength=PAPER_STRENGTH,
            highlight_gamma=PAPER_HIGHLIGHT_GAMMA,
            tile_size=PAPER_TILE_SIZE,
            preserve_pure_white=PAPER_PRESERVE_PURE_WHITE,
        )

    raise ValueError(f"未知 preproc_mode: {mode}")

# ==========================================
# 新增：互動式 Map Layout Polygon 標註工具
# ==========================================
def annotate_map_layout(img_path: str, json_out_path: str):
    """
    打開 UI 讓使用者手動框選 Map Layout (Polygon)，並儲存成 JSON_FORMAT
    """
    img_pil = np.array(Image.open(img_path).convert("RGB"))
    H, W = img_pil.shape[:2]
    
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_pil)
    ax.set_axis_off()
    
    status = ax.text(
        0.01, 0.99, "Draw Layout Polygon | ESC: stop program without saving",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"), color="white"
    )
    
    def onselect(vertices):
        nonlocal H, W
        if len(vertices) < 3:
            return
        poly_pts = [[max(0, min(W-1, int(round(v[0])))), max(0, min(H-1, int(round(v[1]))))] for v in vertices]
        json_content = JSON_FORMAT.format(json.dumps(poly_pts))
        with open(json_out_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        print(f"✅ 已儲存 Layout JSON: {json_out_path}")
        plt.close(fig)
        tornado.ioloop.IOLoop.current().stop()

    def on_key(event):
        if event.key == "escape":
            print("⏭️ 已跳過當前影像的手動標註")
            plt.close(fig)
            if hasattr(tornado.ioloop.IOLoop.current(), 'stop'):
                tornado.ioloop.IOLoop.current().stop()

    ps = PolygonSelector(
        ax, onselect, useblit=False, 
        props=dict(color='lime', linestyle='-', linewidth=2, alpha=0.8)
    )
    
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()  # 會在此處阻斷，等待 user 操作完畢

# ====== 2) 收集檔案 ======
if RECURSIVE:
    files = [p for p in IN_DIR.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
else:
    files = [p for p in IN_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTS]

print(f"IN_DIR : {IN_DIR}")
print(f"OUT_DIR: {OUT_DIR}")
print(f"Files  : {len(files)}")
print(
    f"Params : mode={get_effective_preproc_mode()}, "
    f"ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA}, "
    f"GLOBAL_PRESERVE_PURE_WHITE={GLOBAL_PRESERVE_PURE_WHITE}, "
    f"PAPER_PRESERVE_PURE_WHITE={PAPER_PRESERVE_PURE_WHITE}"
)
# ====== 3) 批次處理 ======
bad = []
for src in tqdm(files):
    rel = src.relative_to(IN_DIR)
    dst = OUT_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(str(src).replace(src.suffix, ".json")):
        annotate_map_layout(str(src), str(src).replace(src.suffix, ".json"))
    layout_geometry = loadLayoutJson(str(src).replace(src.suffix, ".json")).map

    legends = loadLegendJson(
        str(src).replace(config.args.map_dir, config.args.legend_dir).replace(src.suffix, ".json")
    )

    effective_mode = get_effective_preproc_mode()
    legend_subdir = config.args.preproc_mode.replace('none', 'original')

    l_dir = (
        Path(str(src).replace(config.args.map_dir, config.args.legend_dir).replace(src.suffix, ".json")).parent
        / legend_subdir
        / src.stem
    )
    l_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        bad.append(str(src))
        continue
    H, W = img.shape[:2]
    img = apply_preprocess_by_mode(img)

    # 先輸出 legend patch
    for item in legends.features:
        if item.type == MapUnitType.POLYGON:
            min_pt, max_pt = boundingBox(item.label_bbox)
            patch = img[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]]
            ok_patch = cv2.imwrite(l_dir / f"{item.label.replace(' ', config.args.underscore_replace)}.tif", patch)
            if not ok_patch:
                bad.append(f"{src} :: legend_patch::{item.label}")

    # 再做 map layout crop，這段你原本被砍掉了，必須補回來
    if img.ndim == 2:
        img_layout, offset = mask_and_crop(img[None, :, :], layout_geometry)
        img_out = img_layout[0]
    else:
        img_layout, offset = mask_and_crop(img.transpose(2, 0, 1), layout_geometry)
        img_out = img_layout.transpose(1, 2, 0)

    if img_out is None:
        bad.append(str(src))
        continue

    with open(str(dst).replace(dst.suffix, ".offset"), "w") as f:
        f.write(f'{H} {W} {offset[0]} {offset[1]}')

    if os.path.exists(str(dst)):
        print(f"[WARN] 輸出檔已存在，跳過：{dst}")
    else:
        ok = cv2.imwrite(str(dst), img_out)
        if not ok:
            bad.append(str(src))

print("Done.")
if bad:
    print("Failed files (first 10):")
    for x in bad[:10]:
        print(" -", x)