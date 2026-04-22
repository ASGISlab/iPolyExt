# pip install -q geopandas rasterio shapely fiona pyproj pandas opencv-python

import re
from pathlib import Path

import numpy as np
import cv2, sys, os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
from src.augment_and_train_classifier.augment_legend import main as augment_legend_main
current_map = config.args.map
# -------------------------
# 0) 路徑（照你指定）
# -------------------------
map_name=current_map
root_dir = Path(f'data/segmentation_annotation/augmentation/{map_name}_AnnoGT')
seg_dir  = Path(f'{config.args.cls_key_dir}/{map_name}_whitebg_seg')
seg_dir.mkdir(parents=True, exist_ok=True)
mask_dir  = Path(config.args.eval_key_dir)
mask_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1) 找唯一底圖（tif/tiff/png 任一張）
# -------------------------
def find_single_raster(folder: Path) -> Path:
    exts = [".tif", ".tiff", ".png"]
    parent_dir = folder.parent
    stem_name = folder.name
    cands = []
    for e in exts:
        cands += list(parent_dir.glob(f"{stem_name}{e}"))
        cands += list(parent_dir.glob(f"{stem_name}{e.upper()}"))
    cands = sorted({p.resolve() for p in cands})
    if len(cands) != 1:
        raise RuntimeError(
            f"[ERROR] 需要恰好 1 張底圖，但找到 {len(cands)} 張：\n" +
            "\n".join(p.name for p in cands)
        )
    return cands[0]

def read_rgb_any(path: Path) -> np.ndarray:
    """
    讀入底圖並回傳 RGB uint8 (H,W,3)
    - tif/tiff: rasterio
    - png: rasterio 失敗則 cv2 fallback
    """
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

    try:
        with rasterio.open(path) as src:
            arr = src.read()  # (bands,H,W)
            arr_hwc = np.moveaxis(arr, 0, -1)  # (H,W,bands)
            rgb = to_uint8_rgb(arr_hwc)
            # rasterio 讀進來通常已是 RGB band order；若你確定是 BGR 再自行調整
            return rgb
    except Exception:
        bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise RuntimeError(f"[ERROR] 讀底圖失敗：{path}")
        if bgr.ndim == 2:
            rgb = np.repeat(bgr[..., None], 3, axis=2)
        else:
            if bgr.shape[2] == 4:
                bgr = bgr[:, :, :3]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8)

# -------------------------
# 2) 幾何修復（避免 invalid 造成 rasterize 畫不到）
# -------------------------
try:
    from shapely.validation import make_valid as _make_valid  # shapely>=2
    def fix_geom(g):
        try: return _make_valid(g)
        except Exception: return g
except Exception:
    def fix_geom(g):
        try: return g.buffer(0)
        except Exception: return g

# -------------------------
# 3) 讀底圖、找 geojson
# -------------------------
raster_path = find_single_raster(Path(config.args.map_dir)/current_map)
rgb = read_rgb_any(raster_path)
H, W = rgb.shape[:2]
print(f"[BASE] {raster_path.name}  H,W=({H},{W})")

geojsons = sorted(root_dir.glob(f"{map_name}_*.geojson"))
if not geojsons:
    raise FileNotFoundError(f"[ERROR] 找不到 {map_name}_*.geojson：{root_dir}")
print(f"[INFO] geojson 數量 = {len(geojsons)}")

# -------------------------
# 4) 關鍵：用「影像座標」transform（不做 from_bounds、不做 CRS 轉換）
#    你的 geojson y 是負的：y=0 在上，往下變負
#    所以：col=x, row=-y
# -------------------------
transform = Affine(1, 0, 0, 0, -1, 0)  # x=col, y=-row
print(f"[USE] pixel-space transform = {transform}")

# -------------------------
# 5) 逐一輸出 seg.png（白底分割圖）
# -------------------------
label_re = re.compile(rf"^{re.escape(map_name)}_(.+?)\.geojson$", re.I)

for gj in geojsons:
    m = label_re.search(gj.name)
    if not m:
        continue
    label = m.group(1)

    gdf = gpd.read_file(gj)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    if len(gdf) == 0:
        print(f"[SKIP] {gj.name} 幾何空")
        continue

    gdf["geometry"] = gdf["geometry"].apply(fix_geom)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    if len(gdf) == 0:
        print(f"[SKIP] {gj.name} 修復後仍空")
        continue
    
    with open(str(Path(config.args.map_dir)/current_map)+'.offset', 'r') as f:
            h, w, offset_x, offset_y = map(int, f.read().split())
    wholemap_mask=np.zeros((h, w), dtype=bool)
    mask = rasterize(
        shapes=[(geom, 1) for geom in gdf.geometry],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    ).astype(bool)
    wholemap_mask[offset_y:offset_y+H, offset_x:offset_x+W] = mask
    with rasterio.open(mask_dir / f"{map_name}_{label}_poly.tif", 'w', driver='GTiff', compress='lzw', height=h, width=w,
                       count=1, dtype=np.uint8) as fh:
        fh.write(wholemap_mask[None,:,:].astype(np.uint8))
    nz = int(mask.sum())
    cover = nz / float(H*W) * 100.0
    print(f"[OK] {gj.stem}: nonzero={nz} ({cover:.3f}%)")

    seg = np.full((H, W, 3), 255, dtype=np.uint8)
    seg[mask] = rgb[mask]
    out_seg = seg_dir / f"{label}_seg.png"
    if os.path.exists(str(out_seg)):
        print(f"[WARM] {gj.stem} 已存在")
    else:
        cv2.imwrite(str(out_seg), cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))


print(f"\n🎉 完成：{seg_dir}")
