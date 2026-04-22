
from pathlib import Path
import sys
import yaml
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
from src.augment_and_train_classifier.add_blank_classification import main as add_blank_classification_main
current_map = config.args.map
method = config.args.cls_method
cfg    = "configs/augment_and_train_classifier/add_blank_classification.yaml"

overrides = {
    "base_dir": f"{config.args.cls_dir}/stage_2/{method}/{current_map}",
    "num_images": 280 if method=="gen_legend_ratio_augmentation" else 300,#ours 280, geonet 300
}

# 1) load base yaml
with open(cfg, "r", encoding="utf-8") as f:
    d = yaml.safe_load(f)

# 2) apply overrides
d.update(overrides)
add_blank_classification_main(d)

