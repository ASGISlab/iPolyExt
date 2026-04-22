
from pathlib import Path
import sys, subprocess
import yaml
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
from src.augment_and_train_classifier.legend_white_mask import main as legend_white_mask_main

current_map = config.args.map
script = "src/augment_and_train_classifier/legend_white_mask.py"
cfg    = "configs/augment_and_train_classifier/legend_white_mask.yaml"
method = "gen_legend_ratio_augmentation"
overrides = {
    "seed": 1234,
    "src_root": f"{config.args.cls_dir}/stage_1/{method}/{current_map}",
    "out_root": f"{config.args.cls_dir}/stage_2/{method}",  
}

# 1) load base yaml
with open(cfg, "r", encoding="utf-8") as f:
    d = yaml.safe_load(f)

# 2) apply overrides
d.update(overrides)

# 3) apply white mask augmentation controls from outer args
d.setdefault("noise_area_target_single", {})

d["noise_area_target_single"]["target"] = float(config.args.white_mask_area_target)
d["noise_area_target_single"]["min"] = float(config.args.white_mask_area_min)
d["noise_area_target_single"]["max"] = float(config.args.white_mask_area_max)
d["cand_tries_num"] = int(config.args.white_mask_cand_tries_num)

print(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
# 4) run
legend_white_mask_main(d)
