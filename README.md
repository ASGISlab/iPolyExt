# Quickstart

```bash
git clone repo
pip install geopandas rasterio shapely pandas opencv-python tqdm pyyaml Pillow matplotlib PyQt5 scikit-learn ray imagesize
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install timm 
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -i https://test.pypi.org/simple/ cmaas-utils==0.2.1
pip install git+https://github.com/DARPA-CRITICALMAAS/cdr_schemas.git@v0.4.9
git submodule update --init --recursive
git apply --directory=submodules/validation validation.patch
```

To reproduce the GeoNet experiments, you must install TensorFlow, as it is the required framework for the GeoNet classifier scripts:
```bash
pip install tensorflow
```

## Processing Your Custom Map

### Step 1: Data Preparation & Preprocessing
First, ensure your map file (`your_map.[tif/png/...]`) is placed in `/path/to/data/map/`. 
You will also need two precomputed JSON files:
1. **Map Layout Data:** Located at `/path/to/data/map/your_map.json` in the [Uncharted JSON format](https://github.com/DARPA-CRITICALMAAS/uiuc-pipeline#uncharted-json-format). *(If this is missing, you will need to manually annotate the map layout).*
2. **Legend Data:** Located at `/path/to/data/legend/` in the [USGS JSON format](https://github.com/DARPA-CRITICALMAAS/uiuc-pipeline#usgs-json-format).

The preprocessing script will extract the map layout and save the cropped image to `/path/to/data/cropped_map`. It will also extract the legend patches to `/path/to/data/legend/[original/global_consistent/paper_tint]/your_map`. You can manually restore the legend patches afterward if needed.

**Preprocessing Modes (`--preproc_mode`):**
*   `global_consistent`: Recommended for historical maps.
*   `paper_tint`: Recommended for maps with a large number of white pixels.

```bash
python ./scripts/data_preproc.py \
    --map_dir=/path/to/data/map \
    --map_out_dir=/path/to/data/cropped_map \
    --legend_dir=/path/to/data/legend \
    --preproc_mode=global_consistent
```

### Step 2: Create Negative Training Data (Optional)
If your map contains regions that are not represented in the legend (e.g., oceans, lakes, or blank borders), you should manually create bounding boxes for these areas to prevent false positives during extraction. This script will generate training data for the `others` class and save it to `/path/to/data/cls_data`.

```bash
python ./scripts/create_others_training_data.py \
    --map=your_map \
    --map_dir=/path/to/data/cropped_map \
    --cls_dir=/path/to/data/cls_data
```

### Step 3: Generate Legend Training Data
Run the following scripts sequentially to generate, blend, augment, and split the training data for the legend classes. The final dataset will be organized into `/path/to/data/cls_data/stage_3/gen_legend_ratio_augmentation/your_map/apple_pie` *(Note: `apple_pie` is used here as the designated run name)*.

```bash
# 1. Create blank training data patches
python ./scripts/create_blank_training_data.py --map=your_map --cls_dir=/path/to/data/cls_data

# 2. Blend contours into the map patches
python ./scripts/blend_contour_map.py --map=your_map --legend_patch_dir=/path/to/data/legend/global_consistent --cls_dir=/path/to/data/cls_data

# 3. Augment data shapes
python ./scripts/augment_data_shape.py --map=your_map --cls_dir=/path/to/data/cls_data

# 4. Split into training and validation sets
python ./scripts/split_training_val_data.py --map=your_map --cls_dir=/path/to/data/cls_data --run_name=apple_pie
```

### Step 4: Train the Classifier
Train a classifier specifically for your map. The script will locate the training data under the `apple_pie` directory in your `cls_data` folder and save the model checkpoints to `/path/to/model/gen_legend_ratio_augmentation/your_map/apple_pie`.

```bash
python ./scripts/train_cls.py \
    --map=your_map \
    --cls_dir=/path/to/data/cls_data \
    --run_name=apple_pie \
    --model_dir=/path/to/model
```

### Step 5: Polygon Extraction
Finally, use the trained classifier (`apple_pie`) alongside the SAM (Segment Anything Model) to extract the map polygons based on the legend. The output masks will be saved to `/path/to/data/output_dir/apple_pie/gen_legend_ratio_augmentation/output/round_[1~4]`.

**Hardware Requirements:**
*   This step requires GPUs. Set the `--gpu_ids` argument based on your machine (e.g., `--gpu_ids=0` for a single GPU, or `--gpu_ids=0,1` for two GPUs).
*   If you encounter **GPU Out-Of-Memory (OOM)** errors, lower the `--sam_procs` (default: 4) and `--cls_procs` (default: 4) arguments.
*   *Note: The default settings were tested on a machine with 4x V100 GPUs and 504GB of RAM.*

```bash
python ./scripts/polygon_extraction.py \
    --map=your_map \
    --map_dir=/path/to/data/cropped_map \
    --cls_ckpt=apple_pie \
    --out_dir=/path/to/data/output_dir \
    --run_name=apple_pie \
    --model_dir=/path/to/model \
    --gpu_ids=0,1
```


## Reproducing Experiments on TJCP

This section covers the workflow for processing scanned maps, generating training/validation data, and reproducing the experimental results (including the GeoNet baseline).

### Step 1: Data Extraction & Preprocessing
First, extract `data.tar` into the `data` directory located at the root of this repo.
Then, extract `TJCP_models.tar` into the `model` directory located at the root of this repo.


Next, run the preprocessing script. The `--legend_dir` should point to the directory containing precomputed legend data in the USGS JSON format.

**1. With Image Transformation:** Apply image transformations. Cropped maps will be saved to `data/cropped_map_transformed` and legends to `data/Legend/transformed`.
```bash
python ./scripts/data_preproc.py \
    --map_dir=data/map \
    --map_out_dir=data/cropped_map_transformed \
    --legend_dir=data/Legend \
    --preproc_mode=global_consistent
```

**2. Without Image Transformation:** Run the same process without transformations. Cropped maps will be saved to `data/cropped_map` and legends to `data/Legend/original`.
```bash
python ./scripts/data_preproc.py \
    --map_dir=data/map \
    --map_out_dir=data/cropped_map \
    --legend_dir=data/Legend
```

### Step 2: Generate Validation Data (Ground Truth)
Convert the annotation data (GeoJSON) into TIF format to create polygon validation data. This will be used later for evaluating the classifier and polygon extraction. 

```bash
# Generate ground truth TIFFs for all maps based on the transformed cropped maps
python ./scripts/geojson2tiff.py --map=XIII --map_dir=data/cropped_map_transformed
python ./scripts/geojson2tiff.py --map=XIV --map_dir=data/cropped_map_transformed
python ./scripts/geojson2tiff.py --map=13 --map_dir=data/cropped_map_transformed
python ./scripts/geojson2tiff.py --map=17 --map_dir=data/cropped_map_transformed
python ./scripts/geojson2tiff.py --map=35 --map_dir=data/cropped_map_transformed

# Generate "Others" category validation data (Only applies to maps XIII and XIV)
python ./scripts/others_tif.py --map=XIII --map_dir=data/cropped_map_transformed
python ./scripts/others_tif.py --map=XIV --map_dir=data/cropped_map_transformed
```

### Step 3: Generate Training Data (Proposed Method)
Run the following commands sequentially to generate the training dataset, blend contours, apply augmentations, and split the data into training and validation sets. 

*Note: The split data will be saved under `data/train_data/stage_3/gen_legend_ratio_augmentation/<map_name>`, and `my_exp` is used here as the run name.*

```bash
# 1. Generate "Others" category training data (Only for XIII and XIV)
python ./scripts/create_others_training_data.py --map=XIII --map_dir=data/cropped_map_transformed
python ./scripts/create_others_training_data.py --map=XIV --map_dir=data/cropped_map_transformed

# 2. Generate "Blank" category training data
for map_id in 13 17 35 XIII XIV; do
    python ./scripts/create_blank_training_data.py --map=$map_id
done

# 3. Blend contours using manually restored legend patches
for map_id in 13 17 35 XIII XIV; do
    python ./scripts/blend_contour_map.py --map=$map_id --legend_patch_dir=data/Legend/transformed_restored
done

# 4. Augment training data (Randomly removes pixels within the polygon area of patches)
for map_id in 13 17 35 XIII XIV; do
    python ./scripts/augment_data_shape.py --map=$map_id
done

# 5. Split into training and validation sets
for map_id in 13 17 35 XIII XIV; do
    python ./scripts/split_training_val_data.py --map=$map_id --run_name=my_exp --cls_dir=data/train_data
done
```
*(Note: Bash `for` loops are used above to keep the documentation clean. You can also run them line-by-line as in your original script).*

### Step 4: Train Classifiers
Train an individual classifier for each map based on the generated data.
```bash
python ./scripts/train_cls.py --map=13
python ./scripts/train_cls.py --map=17
python ./scripts/train_cls.py --map=35
python ./scripts/train_cls.py --map=XIII
python ./scripts/train_cls.py --map=XIV
```

### Step 5: Evaluate Classifiers
Evaluate the classifiers. The commands below use the provided pre-trained checkpoints (e.g., `fixed_train_data_20260316_173955`). 
**⚠️ Important:** If you want to evaluate the models you just trained yourself, replace the `--cls_ckpt` and `--cls_data` arguments with your specific `run_name` (e.g., `my_exp`).

```bash
# Evaluate Legend classes
python ./scripts/eval_cls.py --map=13 --cls_ckpt=fixed_train_data_20260316_173955 --cls_data=fixed_train_data_20260316_173955 --cls_eval_target=legends --cls_method=gen_legend_ratio_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=17 --cls_ckpt=fixed_train_data_20260316_172943 --cls_data=fixed_train_data_20260316_172943 --cls_eval_target=legends --cls_method=gen_legend_ratio_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=35 --cls_ckpt=fixed_train_data_20260316_172120 --cls_data=fixed_train_data_20260316_172120 --cls_eval_target=legends --cls_method=gen_legend_ratio_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=XIII --cls_ckpt=fixed_train_data_20260316_171401 --cls_data=fixed_train_data_20260316_171401 --cls_eval_target=legends --cls_method=gen_legend_ratio_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=XIV --cls_ckpt=fixed_train_data_20260316_165840 --cls_data=fixed_train_data_20260316_165840 --cls_eval_target=legends --cls_method=gen_legend_ratio_augmentation --cls_key_dir=data/cls_answer_key

# Evaluate Others classes (XIII & XIV only)
python ./scripts/eval_cls.py --map=XIII --cls_ckpt=fixed_train_data_20260316_171401 --cls_data=fixed_train_data_20260316_171401 --cls_eval_target=others --cls_method=gen_legend_ratio_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=XIV --cls_ckpt=fixed_train_data_20260316_165840 --cls_data=fixed_train_data_20260316_165840 --cls_eval_target=others --cls_method=gen_legend_ratio_augmentation --cls_key_dir=data/cls_answer_key

# Generate Evaluation Reports
python ./scripts/cls_eval_report.py --cls_method=gen_legend_ratio_augmentation --cls_eval_target=others
python ./scripts/cls_eval_report.py --cls_method=gen_legend_ratio_augmentation --cls_eval_target=legends
```

***

### Step 6: GeoNet Baseline (Optional)
To reproduce the GeoNet baseline experiments, follow this pipeline to generate data, train, and evaluate using the `GeoNet_Orig_augmentation` method.

```bash
# 1. Generate Training Data
python ./scripts/create_others_training_data.py --map=XIII --cls_method=GeoNet_Orig_augmentation --map_dir=data/cropped_map_transformed
python ./scripts/create_others_training_data.py --map=XIV --cls_method=GeoNet_Orig_augmentation --map_dir=data/cropped_map_transformed

for map_id in 13 17 35 XIII XIV; do
    python ./scripts/create_blank_training_data.py --map=$map_id --cls_method=GeoNet_Orig_augmentation
    python ./scripts/split_training_val_data.py --map=$map_id --cls_method=GeoNet_Orig_augmentation --run_name=my_exp
done

# 2. Train GeoNet Classifiers
for map_id in 13 17 35 XIII XIV; do
    python ./scripts/geonet_train_cls.py --map=$map_id
done

# 3. Evaluate GeoNet Classifiers (Replace ckpt/data variables if using your own models)
python ./scripts/eval_cls.py --map=13 --cls_ckpt=fixed_train_data_20260317_142546 --cls_data=fixed_train_data_20260317_142546 --cls_eval_target=legends --cls_method=GeoNet_Orig_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=17 --cls_ckpt=fixed_train_data_20260317_144221 --cls_data=fixed_train_data_20260317_144221 --cls_eval_target=legends --cls_method=GeoNet_Orig_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=35 --cls_ckpt=fixed_train_data_20260317_150220 --cls_data=fixed_train_data_20260317_150220 --cls_eval_target=legends --cls_method=GeoNet_Orig_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=XIII --cls_ckpt=fixed_train_data_20260317_135735 --cls_data=fixed_train_data_20260317_135735 --cls_eval_target=legends --cls_method=GeoNet_Orig_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=XIV --cls_ckpt=fixed_train_data_20260317_141239 --cls_data=fixed_train_data_20260317_141239 --cls_eval_target=legends --cls_method=GeoNet_Orig_augmentation --cls_key_dir=data/cls_answer_key

python ./scripts/eval_cls.py --map=XIII --cls_ckpt=fixed_train_data_20260317_135735 --cls_data=fixed_train_data_20260317_135735 --cls_eval_target=others --cls_method=GeoNet_Orig_augmentation --cls_key_dir=data/cls_answer_key
python ./scripts/eval_cls.py --map=XIV --cls_ckpt=fixed_train_data_20260317_141239 --cls_data=fixed_train_data_20260317_141239 --cls_eval_target=others --cls_method=GeoNet_Orig_augmentation --cls_key_dir=data/cls_answer_key

# 4. Generate Reports
python ./scripts/cls_eval_report.py --cls_method=GeoNet_Orig_augmentation --cls_eval_target=others
python ./scripts/cls_eval_report.py --cls_method=GeoNet_Orig_augmentation --cls_eval_target=legends
```

***

### Step 7: Polygon Extraction (Reproducing Paper Results)
Use the trained classifiers to reproduce the legend polygon extraction experiments from the paper. The resulting TIFF masks will be saved to `run_seg_cls_loop_result/my_exp/gen_legend_ratio_augmentation/output/round_[1~4]`.

*(Remember to update `--cls_ckpt` to your run name if you are not using the pre-trained checkpoints).*

```bash
python ./scripts/polygon_extraction.py --cls_method=gen_legend_ratio_augmentation --map=XIII --map_dir=data/cropped_map_transformed --cls_ckpt=fixed_train_data_20260316_171401 --out_dir=run_seg_cls_loop_result --run_name=my_exp
python ./scripts/polygon_extraction.py --cls_method=gen_legend_ratio_augmentation --map=XIV --map_dir=data/cropped_map_transformed --cls_ckpt=fixed_train_data_20260316_165840 --out_dir=run_seg_cls_loop_result --run_name=my_exp
python ./scripts/polygon_extraction.py --cls_method=gen_legend_ratio_augmentation --map=13 --map_dir=data/cropped_map_transformed --cls_ckpt=fixed_train_data_20260316_173955 --out_dir=run_seg_cls_loop_result --run_name=my_exp
python ./scripts/polygon_extraction.py --cls_method=gen_legend_ratio_augmentation --map=17 --map_dir=data/cropped_map_transformed --cls_ckpt=fixed_train_data_20260316_172943 --out_dir=run_seg_cls_loop_result --run_name=my_exp
python ./scripts/polygon_extraction.py --cls_method=gen_legend_ratio_augmentation --map=35 --map_dir=data/cropped_map_transformed --cls_ckpt=fixed_train_data_20260316_172120 --out_dir=run_seg_cls_loop_result --run_name=my_exp
```

### Step 8: Evaluate Polygon Extraction
Finally, evaluate the accuracy of the extracted polygons.
```bash
python ./scripts/eval_poly.py \
    -p run_seg_cls_loop_result/my_exp/gen_legend_ratio_augmentation/output/round_4/ \
    -t data/evaluation_answer_key \
    -m data/map \
    -l data/Legend \
    --output run_seg_cls_loop_result/my_exp/gen_legend_ratio_augmentation/output/round_4/feedback/ \
    --feedback
```


## Reproducing Experiments on AI4CMA
This section covers the workflow for reproducing the experimental results using the trained models.

### Step 1: Data Extraction & Preprocessing
First, extract `data.tar` into the `data` directory located at the root of this repo.
Then, extract `AI4CMA_models.tar` into the `model_wo_restoration` directory located at the root of this repo.
You also need to copy `sam_1` directroy from `TJCP_models.tar` into `model_wo_restoration`.

### Step 2: Polygon Extraction (Reproducing Paper Results)
Use the trained classifiers to reproduce the legend polygon extraction experiments from the paper. The resulting TIFF masks will be saved to `ai4cma_result_wo_restoration/no_restoration/gen_legend_ratio_augmentation/output/round_[1~4]`.

```bash
MAP_LIST=(
    "CO_DenverW" "JosCtyOR" "MT_Havre" "MT_RedRockLakes" 
    "OR_JosephineCounty" "OR_VancouverOrchard_basemap" "OR_Washougal" 
    "RI_Uxbridge" "SD_BlackHills" "TX_Driftwood_Wimberley" 
    "USCan_LakeSuperior" "VA_Hayfield" "VA_Stanardsville" 
    "WA_BeaconRock" "WA_DesMoines" "WA_NWSeattle" 
    "WA_PovertyBay2004" "WA_Woodland" "WV_BlueGap" 
    "WY_CO_Peach" "WY_EatonRes" "WY_FortCollins" "WY_LakeOwen"
)

for ((i=0; i<${#MAP_LIST[@]}; i++)); do
    MAP="${MAP_LIST[i]}"
    echo "Processing: $MAP"

    python ./scripts/polygon_extraction.py \
        --cls_method=gen_legend_ratio_augmentation \
        --map="$MAP" \
        --map_dir=data/ai4cma_cropped_map_transformered \
        --cls_ckpt=no_restoration \
        --out_dir=ai4cma_result_wo_restoration \
        --model_dir=model_wo_restoration \
        --run_name=no_restoration
done
```