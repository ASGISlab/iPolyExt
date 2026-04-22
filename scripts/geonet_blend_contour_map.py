import os
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX'
from pathlib import Path
import random
import json, sys, shutil, re
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
from tensorflow.keras.preprocessing.image import ImageDataGenerator
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
from src.augment_and_train_classifier.augment_legend import main as augment_legend_main
current_map = config.args.map
method = "GeoNet_Orig_augmentation"
cfg    = "configs/augment_and_train_classifier/augment_legend.yaml"

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            return img.convert("RGBA")
    except IOError:
        print(f"Unable to open image: {image_path}")
        return None

def is_image_white(image):
    arr = np.array(image)
    return np.all(arr[:, :, :3] >= 250)

def make_image_translucent(image, opacity):
    translucent_image = Image.new("RGBA", image.size)
    translucent_image.paste(image, (0, 0), image.split()[3].point(lambda p: p * opacity // 255))
    return translucent_image

def apply_translucent_overlay(large_image_path, small_image, small_image_path, save_dir, iterations=100, opacity_values=[0.25, 0.5, 0.75]):
    if small_image is None:
        return
    with Image.open(large_image_path) as img:
        large_image = img.convert("RGBA")

        if large_image.width < small_image.width or large_image.height < small_image.height:
            print(f"Large image {large_image_path} is smaller than small image {small_image_path}, skipping.")
            return

        small_image_label = os.path.splitext(os.path.basename(small_image_path))[0]
        for opacity in opacity_values:
            translucent_large_image = make_image_translucent(large_image, int(opacity * 255))
            for j in range(iterations):
                x = random.randint(0, translucent_large_image.width - small_image.width)
                y = random.randint(0, translucent_large_image.height - small_image.height)
                crop = translucent_large_image.crop((x, y, x + small_image.width, y + small_image.height))
                enhancer = ImageEnhance.Color(crop)
                crop = enhancer.enhance(random.uniform(0.5, 1.5))

                if not is_image_white(crop):
                    combined_image = Image.alpha_composite(small_image, crop)

                    """# Data augmentation
                    data_augmentation = ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest'
                    )"""
                    # Data augmentation with NO transformations
                    data_augmentation = ImageDataGenerator()

                    combined_image_array = np.expand_dims(np.array(combined_image), axis=0)
                    it = data_augmentation.flow(
                        combined_image_array,
                        batch_size=1,
                        shuffle=False,
                        seed=SEED
                    )
                    augmented_image = next(it)
                    augmented_image_pil = Image.fromarray(augmented_image[0].astype('uint8'), 'RGBA')

                    variant_dir = os.path.join(save_dir, small_image_label, f"opacity_{int(opacity*100)}", str(j))
                    os.makedirs(variant_dir, exist_ok=True)
                    filename = f"{small_image_label}.png"
                    augmented_image_pil.save(os.path.join(variant_dir, filename))
                    print(f"Saved: {os.path.join(variant_dir, filename)}")

                    metadata = {
                        'filename': filename,
                        'label': small_image_label,
                        'opacity': opacity,
                        'crop_position': (x, y)
                    }
                    with open(os.path.join(variant_dir, f"{small_image_label}.json"), 'w') as f:
                        json.dump(metadata, f)

def process_folder(small_images_folder, large_image_path, save_dir, opacity_values=[0.25, 0.5, 0.75]):
    os.makedirs(save_dir, exist_ok=True)
    for image_name in sorted(os.listdir(small_images_folder)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            small_image_path = os.path.join(small_images_folder, image_name)
            small_image = load_image(small_image_path)
            if small_image:
                apply_translucent_overlay(large_image_path, small_image, small_image_path, save_dir, opacity_values=opacity_values)

def process_all_folders(top_folder, large_image_path, base_save_dir, opacity_values=[0.25, 0.5, 0.75]):
    os.makedirs(base_save_dir, exist_ok=True)
    for folder_name in sorted(os.listdir(top_folder)):
        folder_path = os.path.join(top_folder, folder_name)
        if os.path.isdir(folder_path):
            save_dir = os.path.join(base_save_dir, folder_name)
            os.makedirs(save_dir, exist_ok=True)
            process_folder(folder_path, large_image_path, save_dir, opacity_values=opacity_values)


large_image_path = f"contour_map/地理圖已處理/contour_sample.png"
# top_folder = 'images/output/removed_text'
top_folder = f"data/Legend/augmentation_Legend_processed_Frameless/{current_map}"
base_save_dir = f"data/train_data/stage_1/{method}/{current_map}"
opacity_values = [0.25, 0.5, 0.75]  # Adjust the opacity values as needed
process_folder(top_folder, large_image_path, base_save_dir, opacity_values)


# ===== 你只要改這一行：輸入 stage_1 根目錄 =====
IN_ROOT = Path(f"data/train_data/stage_1/{method}/{current_map}")

# ===== 輸出 stage_2 根目錄：自動把 stage_1 替換成 stage_2 =====
OUT_ROOT = Path(str(IN_ROOT).replace("/stage_1/", "/stage_2/", 1))

# ===== 行為選項 =====
MOVE_INSTEAD_OF_COPY = False  # True = 移動(搬走)；False = 複製(保留原檔)
VALID_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

assert IN_ROOT.is_dir(), f"[ERROR] IN_ROOT 不存在或不是資料夾：{IN_ROOT}"

opacity_re = re.compile(r"^opacity_(\d+)$")

copied = 0
skipped = 0
warnings = 0

print("[IN ]", IN_ROOT)
print("[OUT]", OUT_ROOT)

for label_dir in sorted([p for p in IN_ROOT.iterdir() if p.is_dir()]):
    label = label_dir.name  # 例如 Ml
    out_label_dir = OUT_ROOT / label
    out_label_dir.mkdir(parents=True, exist_ok=True)

    # 找 opacity_xx
    for op_dir in sorted([p for p in label_dir.iterdir() if p.is_dir()]):
        m = opacity_re.match(op_dir.name)
        if not m:
            continue

        op_num = int(m.group(1))  # 例如 75
        op_str = f"{op_num:03d}"  # => 075

        # 找像 84 這種 index 資料夾
        for idx_dir in sorted([p for p in op_dir.iterdir() if p.is_dir()]):
            idx = idx_dir.name
            if not idx.isdigit():
                continue

            # 取裡面的圖片檔（可能同時有 .png 和 .json；只處理圖片）
            img_files = [f for f in idx_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXT]
            if not img_files:
                skipped += 1
                continue

            for img_path in sorted(img_files):
                # 你指定的命名規則：<label>__op-075__84.png
                # 若原本不是 label.png，也仍然強制用 label（比較符合你要的整理方式）
                out_name = f"{label}__op-{op_str}__{idx}{img_path.suffix}"
                out_path = out_label_dir / out_name

                # 避免覆蓋：若重名就自動加尾碼
                if out_path.exists():
                    warnings += 1
                    k = 2
                    while True:
                        alt = out_label_dir / f"{label}__op-{op_str}__{idx}__v{k}{img_path.suffix}"
                        if not alt.exists():
                            out_path = alt
                            break
                        k += 1

                if MOVE_INSTEAD_OF_COPY:
                    shutil.move(str(img_path), str(out_path))
                else:
                    shutil.copy2(str(img_path), str(out_path))

                copied += 1

print(f"\nDone. copied={copied}, skipped(no image)={skipped}, warnings(duplicate name)={warnings}")
print("[OUT ROOT]", OUT_ROOT)