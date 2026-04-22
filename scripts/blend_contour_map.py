
from pathlib import Path
import sys
import yaml
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
from src.augment_and_train_classifier.augment_legend import main as augment_legend_main
current_map = config.args.map
method = "gen_legend_ratio_augmentation"
cfg    = "configs/augment_and_train_classifier/augment_legend.yaml"

overrides = {
    "seed": 1234,
    "big_root": config.args.contour_map_dir,
    "small_root": config.args.legend_patch_dir+"/"+current_map,  
    "out_root": f"{config.args.cls_dir}/stage_1/{method}",
}

# 1) load base yaml
with open(cfg, "r", encoding="utf-8") as f:
    d = yaml.safe_load(f)

# 2) apply overrides
d.update(overrides)
augment_legend_main(d)

