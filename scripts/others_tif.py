# ========== BGC 產生器（以 SEG_DIR 尺寸為準，左上角對齊裁切/補白）==========
import numpy as np
import cv2, sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
#this file is for generating others masks for Geological Maps of Taiwan during the Japanese Colonial Period
# ====== 0) 你只要改這裡 ======

MAP_NAME = config.args.map
SEG_DIR  = Path(config.args.cls_key_dir)/f'{MAP_NAME}_whitebg_seg'
ORIG_MAP = Path(f"{config.args.map_dir}/{MAP_NAME}.tif")             # 原始地圖（tif/tiff/png）
                                                # 只用在輸出命名
OUT_BGC_SEG  = SEG_DIR / f"BGC_seg.png"

SEG_BG = 255                                                        # seg 背景色：白色
ORIG_WHITE_TH = 255                                               # 原圖純白判定：三通道皆 255
EXCLUDE_ORIG_WHITE_FROM_BGC = True                                # True：排除原圖純白區（避免混到 blank）

VALID_EXTS = {".png", ".tif", ".tiff"}
EXCLUDE_NAME_KEYWORDS = ["bgc", "bg_color", "background", "blank", ".ipynb_checkpoints"]
REQUIRE_SUFFIX = None  # 例如 "_seg.png"；不限制就 None

# ====== 1) 小工具：讀圖成 RGB uint8 ======
def read_rgb_any(path: Path) -> np.ndarray:
    def to_uint8_rgb(arr_hwc: np.ndarray) -> np.ndarray:
        if arr_hwc.ndim == 2:
            arr_hwc = arr_hwc[..., None]
        if arr_hwc.shape[2] == 1:
            rgb = np.repeat(arr_hwc, 3, axis=2)
        else:
            rgb = arr_hwc[..., :3]
        if rgb.dtype == np.uint8:
            return rgb
        if rgb.dtype == np.uint16:
            return (rgb >> 8).astype(np.uint8)
        if np.issubdtype(rgb.dtype, np.floating):
            mx = float(np.nanmax(rgb)) if rgb.size else 1.0
            if mx <= 1.5:
                rgb = np.clip(rgb, 0.0, 1.0) * 255.0
            else:
                rgb = np.clip(rgb, 0.0, 255.0)
            return rgb.astype(np.uint8)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    # tif/tiff 優先 rasterio
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read()                      # (bands,H,W)
                arr_hwc = np.moveaxis(arr, 0, -1)     # (H,W,bands)
                return to_uint8_rgb(arr_hwc)
        except Exception:
            pass

    # fallback cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"[ERROR] 讀圖失敗：{path}")
    if bgr.ndim == 2:
        rgb = np.repeat(bgr[..., None], 3, axis=2)
    else:
        if bgr.shape[2] == 4:
            bgr = bgr[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)

# 左上角對齊：把 img 變成 (H,W,3)，大就裁切，小就補白
def fit_top_left_rgb(img_rgb: np.ndarray, H: int, W: int, fill=255) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    out = np.full((H, W, 3), fill, dtype=np.uint8)
    hh = min(H, h)
    ww = min(W, w)
    out[:hh, :ww] = img_rgb[:hh, :ww, :3]  # 左上角貼上
    return out

# ====== 2) 蒐集 seg 圖 ======
seg_files = []
for p in SEG_DIR.rglob("*seg.png"):
    if not p.is_file():
        continue
    if p.suffix.lower() not in VALID_EXTS:
        continue
    name_low = p.name.lower()

    if REQUIRE_SUFFIX is not None and (not name_low.endswith(REQUIRE_SUFFIX.lower())):
        continue
    if any(k in name_low for k in EXCLUDE_NAME_KEYWORDS):
        continue
    if p.resolve() == ORIG_MAP.resolve():
        continue

    seg_files.append(p)

seg_files = sorted(seg_files)
if len(seg_files) == 0:
    raise RuntimeError(f"[ERROR] 在 {SEG_DIR} 找不到可用的 seg 圖（檢查 VALID_EXTS / EXCLUDE_NAME_KEYWORDS / REQUIRE_SUFFIX）")

print(f"[INFO] seg 圖數量 = {len(seg_files)}")
print(f"[INFO] 參考尺寸（取第一張） = {seg_files[0].name}")

# ====== 3) 以 SEG_DIR 第一張 seg 圖尺寸作為基準 ======
ref0 = read_rgb_any(seg_files[0])
H_ref, W_ref = ref0.shape[:2]
print(f"[REF] H,W = ({H_ref},{W_ref})  <- 以 SEG_DIR 為準")

# ====== 4) 讀原圖，左上角裁切/補白到 ref 尺寸 ======
orig_rgb_full = read_rgb_any(ORIG_MAP)
orig_rgb = fit_top_left_rgb(orig_rgb_full, H_ref, W_ref, fill=ORIG_WHITE_TH)
print(f"[BASE] ORIG={ORIG_MAP.name} 原尺寸={orig_rgb_full.shape[:2]} -> 對齊後={orig_rgb.shape[:2]}")

# ====== 5) union 其他類別的非白區 ======
union_mask = np.zeros((H_ref, W_ref), dtype=bool)

def nonwhite_mask(img_rgb: np.ndarray) -> np.ndarray:
    return ~np.all(img_rgb == SEG_BG, axis=2)
for i, fp in enumerate(seg_files, 1):
    seg_rgb_full = read_rgb_any(fp)

    seg_rgb = fit_top_left_rgb(seg_rgb_full, H_ref, W_ref, fill=SEG_BG)
    if seg_rgb_full.shape[:2] != (H_ref, W_ref):
        print(f"[WARN] 尺寸不一致：{fp.name} {seg_rgb_full.shape[:2]} -> 已用左上角裁切/補白對齊到 {(H_ref,W_ref)}")

    union_mask |= nonwhite_mask(seg_rgb)
    if i % 50 == 0 or i == len(seg_files):
        print(f"  processed {i}/{len(seg_files)}")

covered = int(union_mask.sum())
print(f"[STAT] union 覆蓋像素 = {covered} / {H_ref*W_ref} ({covered/(H_ref*W_ref)*100:.3f}%)")

# ====== 6) BGC = ref 畫布 - union（可排除原圖純白）=====
bgc_mask = ~union_mask

if EXCLUDE_ORIG_WHITE_FROM_BGC:
    orig_nonwhite_mask = ~np.all(orig_rgb == ORIG_WHITE_TH, axis=2)
    bgc_mask &= orig_nonwhite_mask   # 排除原圖純白，避免把 blank 混進來
    print("[INFO] 已排除原圖純白區（避免 blank 混入 BGC）")
bgc_pixels = int(bgc_mask.sum())
print(f"[STAT] BGC 像素 = {bgc_pixels} / {H_ref*W_ref} ({bgc_pixels/(H_ref*W_ref)*100:.3f}%)")

# ====== 7) 輸出：黑底 + 保留原圖樣式（只在 BGC 區）=====
out_bgc = np.full((H_ref, W_ref, 3), 255, dtype=np.uint8)
out_bgc[bgc_mask] = orig_rgb[bgc_mask]

OUT_BGC_SEG.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT_BGC_SEG), cv2.cvtColor(out_bgc, cv2.COLOR_RGB2BGR))
print(f"[SAVE] BGC seg  -> {OUT_BGC_SEG}")


