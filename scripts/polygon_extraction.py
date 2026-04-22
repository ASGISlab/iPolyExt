from pathlib import Path
import sys, subprocess, copy, time
#root_dir = Path(__file__).resolve().parent.parent
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from src import config
from src.run_seg_cls_loop.seg_round1 import main as seg_round1_main
from src.run_seg_cls_loop.seg_multi_round import main as seg_multi_round_main
from src.run_seg_cls_loop.classify_round1 import main as classify_round1_main
from src.run_seg_cls_loop.classify_multi_round import main as classify_multi_round_main
from src.run_seg_cls_loop.check_seg_cls_round_integrity import main as check_seg_cls_round_integrity_main
from datetime import datetime
import yaml
import ray
#root_dir = Path(__file__).resolve().parent.parent
root_dir = Path(__file__).parent.parent
ray.init('local', _temp_dir=str(root_dir/'tmp'), ignore_reinit_error=True)
Map_number = config.args.map
img_path = f"{config.args.map_dir}/{Map_number}.tif"
cls_ckpt = config.args.cls_ckpt
train_model_method = method = config.args.cls_method
blank_white_area_ratio = config.args.blank_white_area_ratio
small_mask_extra_width = config.args.small_mask_extra_width
# =========================
# 0) 只改這裡
# =========================
PYTHON_BIN = sys.executable

TOTAL_ROUNDS = 4
START_ROUND = config.args.start_round

if not (1 <= START_ROUND <= TOTAL_ROUNDS):
    raise ValueError(f"start_round 必須介於 1 和 {TOTAL_ROUNDS}，目前是 {START_ROUND}")

sam_ckpt = f"{config.args.model_dir}/sam_1/sam_vit_h_4b8939.pth"
SEG_IN_ROOT_BASE = Path(f"{config.args.out_dir}/{config.args.run_name}/sam1") / train_model_method / Map_number
CLASSIFY_OUTPUT_ROOT =Path(f"{config.args.out_dir}/{config.args.run_name}/classify/SAM_post_classify_out")
TIF_OUTPUT_ROOT =Path(f"{config.args.out_dir}/{config.args.run_name}/{train_model_method}/output")
CKPT_PATH =Path(config.args.model_dir) / method / Map_number / cls_ckpt / "best_model.pth"

def parse_float_list(s, expected_len=4):
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(vals)}: {s}")
    return vals

def parse_int_list(s, expected_len=4):
    vals = [int(x.strip()) for x in s.split(",")]
    if len(vals) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(vals)}: {s}")
    return vals

SAM_PRED_IOU_BY_ROUND = parse_float_list(config.args.sam_pred_iou_thresh_by_round)
SAM_STABILITY_BY_ROUND = parse_float_list(config.args.sam_stability_score_thresh_by_round)

VOTE_MAJORITY_TH_BY_ROUND = parse_float_list(config.args.vote_majority_th_by_round)
VOTE_MIN_PATCH_BY_ROUND = parse_int_list(config.args.vote_min_patch_by_round)
VOTE_MIN_NONBLANK_BY_ROUND = parse_int_list(config.args.vote_min_nonblank_by_round)
VOTE_STRONG_MIN_BY_ROUND = parse_int_list(config.args.vote_strong_min_by_round)

# =========================
# 顯示每輪設定，確認有吃到 config.py
# =========================
print("\n" + "=" * 100)
print("[ROUND HYPERPARAMETERS]")
print("=" * 100)

for round_idx in range(1, TOTAL_ROUNDS + 1):
    i = round_idx - 1

    print(f"\n[Round {round_idx}]")

    print("  SAM:")
    print(f"    pred_iou_thresh        = {SAM_PRED_IOU_BY_ROUND[i]}")
    print(f"    stability_score_thresh = {SAM_STABILITY_BY_ROUND[i]}")

    print("  Vote:")
    print(f"    majority_th  = {VOTE_MAJORITY_TH_BY_ROUND[i]}")
    print(f"    min_patch    = {VOTE_MIN_PATCH_BY_ROUND[i]}")
    print(f"    min_nonblank = {VOTE_MIN_NONBLANK_BY_ROUND[i]}")
    print(f"    strong_min   = {VOTE_STRONG_MIN_BY_ROUND[i]}")

print("=" * 100 + "\n")

SCRIPT_SPECS = {
    "seg_round1": {
        "script":seg_round1_main,
        "cfg":"configs/run_seg_cls_loop/seg_round1.yaml",
        "arg": "--config",
    },
    "seg_multi_round": {
        "script":seg_multi_round_main,
        "cfg":"configs/run_seg_cls_loop/seg_multi_round.yaml",
        "arg": "--config",
    },
    "classify_round1": {
        "script":classify_round1_main,
        "cfg":"configs/run_seg_cls_loop/classify_round1.yaml",
        "arg": "--config",
    },
    "classify_multi_round": {
        "script":classify_multi_round_main,
        "cfg":"configs/run_seg_cls_loop/classify_multi_round.yaml",
        "arg": "--config",
    },
    "check_seg_cls_round_integrity": {
        "script":check_seg_cls_round_integrity_main,
        "cfg":"configs/run_seg_cls_loop/check_seg_cls_round_integrity.yaml",
        "arg": "--config",
    },
}

RUN_PLAN = {
    1: ["seg_round1", "classify_round1", "check_seg_cls_round_integrity"],
    2: ["seg_multi_round", "classify_multi_round", "check_seg_cls_round_integrity"],
    3: ["seg_multi_round", "classify_multi_round", "check_seg_cls_round_integrity"],
    4: ["seg_multi_round", "classify_multi_round", "check_seg_cls_round_integrity"],
}

# =========================
# 1) 工具
# =========================
def get_module_group(module_name: str) -> str:
    if module_name.startswith("seg_"):
        return "seg"
    elif module_name.startswith("classify_"):
        return "classify"
    elif module_name.startswith("check_"):
        return "integrity_check"
    return "other"

def deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)

def run_one(module_name: str, round_idx: int, overrides: dict) -> float:
    spec = SCRIPT_SPECS[module_name]
    cfg = load_yaml(spec["cfg"])
    cfg = deep_update(copy.deepcopy(cfg), overrides)

    print("\n" + "=" * 100)
    print(f"[ROUND {round_idx}] {module_name}")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    t0 = time.perf_counter()
    spec["script"](cfg)
    elapsed = time.perf_counter() - t0

    print(f"[DONE] {module_name} | elapsed = {elapsed:.2f} sec ({elapsed/60:.2f} min)")
    return elapsed

# =========================
# 2) 每輪每支要改的值
#    base yaml 不變的就不要寫
# =========================
OVERRIDES = {
    1: {
        "seg_round1": {
            "basic": {
                "train_model_method": train_model_method,
                "Map_number": Map_number,
                "ROUND": 1,
                "num_workers": config.args.num_workers
            },
            "paths": {
                "img_path": img_path,
                "sam_ckpt": sam_ckpt,
                "model_typ": "vit_h",
                "round_root": f"{config.args.out_dir}/{config.args.run_name}/sam1/{train_model_method}/{Map_number}/round_1",
            },
            "sam": {
                "sam_points_per_batch": config.args.sam_bs,
                "sam_pred_iou_thresh": SAM_PRED_IOU_BY_ROUND[0],
                "sam_stability_score_thresh": SAM_STABILITY_BY_ROUND[0],
                "sam_procs": config.args.sam_procs,
                "gpu_ids": config.args.gpu_ids,
            },
            # 這裡補 seg_round1.yaml 需要改的欄位
            # "paths": {...},
            # "sam": {...},
        },
        "classify_round1": {
            "model": {
                "batch_size": config.args.cls_bs,
                "cls_procs": config.args.cls_procs,
                "gpu_ids": config.args.gpu_ids,
            },
            
            "task": {
                "map_number": Map_number,
                "map_dir": config.args.map_dir,
                "round": 1,
                "data_name": cls_ckpt,
                "method": method,
                "train_model_method": train_model_method,
                "num_workers": config.args.num_workers,
                "num_preload": config.args.num_preload,
                'underscore_replace': config.args.underscore_replace
            },
            "paths": {
                "in_roots": [
                    str(SEG_IN_ROOT_BASE / "round_1")
                ],
                "output_root": str(CLASSIFY_OUTPUT_ROOT),
                "ckpt_path": str(CKPT_PATH),
                "tif_output_dir": str(TIF_OUTPUT_ROOT / "round_1"),
                "run_name": config.args.run_name
            },
            "vote": {
                "majority_th": VOTE_MAJORITY_TH_BY_ROUND[0],
                "min_patch": VOTE_MIN_PATCH_BY_ROUND[0],
                "min_nonblank": VOTE_MIN_NONBLANK_BY_ROUND[0],
                "strong_min": VOTE_STRONG_MIN_BY_ROUND[0],
            },
            "blank_rule": {
                "blank_white_area_ratio" : blank_white_area_ratio,
                "small_threshold_extra_w": small_mask_extra_width,
            },
            # 這裡加 round1 特別要改的值
            # "vote": {"min_nonblank": 2},
        },
        "check_seg_cls_round_integrity": {
            "vars": {
                "train_model_method": train_model_method,
                "map_number": Map_number,
                "round": 1,
            },
            "paths": {
                "error_src_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_1" / "segmentation_rejected"),
                "error_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_1" / "round_integrity_check" / "error_only_full"),

                "correct_masks_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_1" / "segmentation_verified" / "merged"),
                "alt_image": img_path,
                "correct_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_1" / "round_integrity_check" / "correct_only_full"),
                
                "union_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_1" / "round_integrity_check" / "error_correct_union_full"),
            },
            # 這裡補 check yaml 需要改的欄位
            # "paths": {...},
        },
    },
    
    2: {
        "seg_multi_round": {
            "task": {
                "train_model_method" : train_model_method,
                "map_number": Map_number,
                "round": 2,
                "num_workers": config.args.num_workers
            },
            "paths": {
                "err_root_template" : str(CLASSIFY_OUTPUT_ROOT / train_model_method / Map_number / "round_1" / "segmentation_rejected" ),
                "out_root_template" : f"{config.args.out_dir}/{config.args.run_name}/sam1/{train_model_method}/{Map_number}/round_2",
                "sam_ckpt": sam_ckpt,
            },
            "model" : {
                "model_type" : "vit_h",
            },
            "sam_generator": {
                "points_per_side": 32,
                "points_per_batch": config.args.sam_bs,
                "pred_iou_thresh": SAM_PRED_IOU_BY_ROUND[1],
                "stability_score_thresh": SAM_STABILITY_BY_ROUND[1],
                "sam_procs": config.args.sam_procs,
                "gpu_ids": config.args.gpu_ids,
            }, 
            # "paths": {...},
        },
        "classify_multi_round": {
            "experiment": {
                "map_number": Map_number,
                "map_dir": config.args.map_dir,
                "round": 2,
                "data_name": cls_ckpt,
                "method": method,
                "train_model_method": train_model_method,
                "map_name": Map_number,
                "num_workers": config.args.num_workers,
                "num_preload": config.args.num_preload,
                'underscore_replace': config.args.underscore_replace
            },
            "paths": {
                "legacy_cell_root": str(f"{config.args.out_dir}"),
                "project_root": str(''),
                "ckpt_path": str(CKPT_PATH),
                "tif_output_dir": str(TIF_OUTPUT_ROOT / "round_2"),
                "run_name": config.args.run_name
            },
            "vote":{
                "majority_th": VOTE_MAJORITY_TH_BY_ROUND[1],
                "min_patch": VOTE_MIN_PATCH_BY_ROUND[1],
                "min_nonblank": VOTE_MIN_NONBLANK_BY_ROUND[1],
                "strong_min": VOTE_STRONG_MIN_BY_ROUND[1],
            },
            "model": {
                "batch_size": config.args.cls_bs,
                "cls_procs": config.args.cls_procs,
                "gpu_ids": config.args.gpu_ids,
            },
            "rules": {
                "blank_white_area_ratio" : blank_white_area_ratio,
                "small_threshold_extra_cols": small_mask_extra_width,
            },
            # round2 特殊調整寫這
            # "vote": {"min_nonblank": 3},
        },        
        "check_seg_cls_round_integrity": {
            "vars": {
                "train_model_method": train_model_method,
                "map_number": Map_number,
                "round": 2,
            },
            "paths": {
                "error_src_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_2" / "segmentation_rejected"),
                "error_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_2" / "round_integrity_check" / "error_only_full"),

                "correct_masks_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_2" / "segmentation_verified" / "merged"),
                "alt_image": img_path,
                "correct_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_2" / "round_integrity_check" / "correct_only_full"),
                
                "union_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_2" / "round_integrity_check" / "error_correct_union_full"),
            },
            # 這裡補 check yaml 需要改的欄位
            # "paths": {...},
        },
    },
    3: {
        "seg_multi_round": {
            "task": {
                "train_model_method" : train_model_method,
                "map_number": Map_number,
                "round": 3,
                "num_workers": config.args.num_workers
            },
            "paths": {
                "err_root_template" : str(CLASSIFY_OUTPUT_ROOT / train_model_method / Map_number / "round_2" / "segmentation_rejected" ),
                "out_root_template" : f"{config.args.out_dir}/{config.args.run_name}/sam1/{train_model_method}/{Map_number}/round_3",
                "sam_ckpt": sam_ckpt,
            },
            "model" : {
                "model_type" : "vit_h",
            },
            "sam_generator": {
                "points_per_side": 32,
                "points_per_batch": config.args.sam_bs,
                "pred_iou_thresh": SAM_PRED_IOU_BY_ROUND[2],
                "stability_score_thresh": SAM_STABILITY_BY_ROUND[2],
                "sam_procs": config.args.sam_procs,
                "gpu_ids": config.args.gpu_ids,
            }, 
            # "paths": {...},
        },
        "classify_multi_round": {
            "experiment": {
                "map_number": Map_number,
                "map_dir": config.args.map_dir,
                "round": 3,
                "data_name": cls_ckpt,
                "method": method,
                "train_model_method": train_model_method,
                "map_name": Map_number,
                "num_workers": config.args.num_workers,
                "num_preload": config.args.num_preload,
                'underscore_replace': config.args.underscore_replace
            },
            "paths": {
                "tif_output_dir": str(TIF_OUTPUT_ROOT / "round_3"),
                "legacy_cell_root": str(f"{config.args.out_dir}"),
                "ckpt_path": str(CKPT_PATH),
                "project_root": str(''),
                "run_name": config.args.run_name
            },
            "vote":{
                "majority_th": VOTE_MAJORITY_TH_BY_ROUND[2],
                "min_patch": VOTE_MIN_PATCH_BY_ROUND[2],
                "min_nonblank": VOTE_MIN_NONBLANK_BY_ROUND[2],
                "strong_min": VOTE_STRONG_MIN_BY_ROUND[2],
            },
            "model": {
                "batch_size": config.args.cls_bs,
                "cls_procs": config.args.cls_procs,
                "gpu_ids": config.args.gpu_ids,
            },
            "rules": {
                "blank_white_area_ratio" : blank_white_area_ratio,
                "small_threshold_extra_cols": small_mask_extra_width,
            },
            # round2 特殊調整寫這
            # "vote": {"min_nonblank": 3},
        },        
        "check_seg_cls_round_integrity": {
            "vars": {
                "train_model_method": train_model_method,
                "map_number": Map_number,
                "round": 3,
            },
            "paths": {
                "error_src_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_3" / "segmentation_rejected"),
                "error_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_3" / "round_integrity_check" / "error_only_full"),

                "correct_masks_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_3" / "segmentation_verified" / "merged"),
                "alt_image": img_path,
                "correct_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_3" / "round_integrity_check" / "correct_only_full"),
                
                "union_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_3" / "round_integrity_check" / "error_correct_union_full"),
            },
            # 這裡補 check yaml 需要改的欄位
            # "paths": {...},
        },
    },
    4: {
        "seg_multi_round": {
            "task": {
                "train_model_method" : train_model_method,
                "map_number": Map_number,
                "round": 4,
                "num_workers": config.args.num_workers
            },
            "paths": {
                "err_root_template" : str(CLASSIFY_OUTPUT_ROOT / train_model_method / Map_number / "round_3" / "segmentation_rejected" ),
                "out_root_template" : f"{config.args.out_dir}/{config.args.run_name}/sam1/{train_model_method}/{Map_number}/round_4",
                "sam_ckpt": sam_ckpt,
            },
            "model" : {
                "model_type" : "vit_h",
            },
            "sam_generator": {
                "points_per_side": 32,
                "points_per_batch": config.args.sam_bs,
                "pred_iou_thresh": SAM_PRED_IOU_BY_ROUND[3],
                "stability_score_thresh": SAM_STABILITY_BY_ROUND[3],
                "sam_procs": config.args.sam_procs,
                "gpu_ids": config.args.gpu_ids,
            }, 
            # "paths": {...},
        },
        "classify_multi_round": {
            "experiment": {
                "map_number": Map_number,
                "map_dir": config.args.map_dir,
                "round": 4,
                "data_name": cls_ckpt,
                "method": method,
                "train_model_method": train_model_method,
                "map_name": Map_number,
                "num_workers": config.args.num_workers,
                "num_preload": config.args.num_preload,
                'underscore_replace': config.args.underscore_replace
            },
            "paths": {
                "ckpt_path": str(CKPT_PATH),
                "legacy_cell_root": str(f"{config.args.out_dir}"),
                "project_root": str(''),
                "tif_output_dir": str(TIF_OUTPUT_ROOT / "round_4"),
                "run_name": config.args.run_name
            },
            "vote":{
                "majority_th": VOTE_MAJORITY_TH_BY_ROUND[3],
                "min_patch": VOTE_MIN_PATCH_BY_ROUND[3],
                "min_nonblank": VOTE_MIN_NONBLANK_BY_ROUND[3],
                "strong_min": VOTE_STRONG_MIN_BY_ROUND[3],
            },
            "model": {
                "batch_size": config.args.cls_bs,
                "cls_procs": config.args.cls_procs,
                "gpu_ids": config.args.gpu_ids,
            },
            "rules": {
                "blank_white_area_ratio" : blank_white_area_ratio,
                "small_threshold_extra_cols": small_mask_extra_width,
            },
            # round2 特殊調整寫這
            # "vote": {"min_nonblank": 3},
        },        
        "check_seg_cls_round_integrity": {
            "vars": {
                "train_model_method": train_model_method,
                "map_number": Map_number,
                "round": 4,
            },
            "paths": {
                "error_src_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_4" / "segmentation_rejected"),
                "error_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_4" / "round_integrity_check" / "error_only_full"),

                "correct_masks_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_4" / "segmentation_verified" / "merged"),
                "alt_image": img_path,
                "correct_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_4" / "round_integrity_check" / "correct_only_full"),
                
                "union_out_dir": str(CLASSIFY_OUTPUT_ROOT/ train_model_method / Map_number / "round_4" / "round_integrity_check" / "error_correct_union_full"),
            },
            # 這裡補 check yaml 需要改的欄位
            # "paths": {...},
        },
    },
}

overall_t0 = time.perf_counter()
started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

time_records = []
round_summaries = []
# =========================
# 3) 跑全部
# =========================
for round_idx in range(START_ROUND, TOTAL_ROUNDS + 1):
    print("\n" + "#" * 100)
    print(f"ROUND {round_idx}")
    print("#" * 100)

    round_t0 = time.perf_counter()
    round_seg_sec = 0.0
    round_classify_sec = 0.0
    round_check_sec = 0.0

    for module_name in RUN_PLAN[round_idx]:
        overrides = OVERRIDES[round_idx].get(module_name, {})
        elapsed = run_one(module_name, round_idx, overrides)

        group = get_module_group(module_name)
        if group == "seg":
            round_seg_sec += elapsed
        elif group == "classify":
            round_classify_sec += elapsed
        elif group == "integrity_check":
            round_check_sec += elapsed

        time_records.append({
            "round": round_idx,
            "module_name": module_name,
            "module_group": group,
            "elapsed_sec": round(elapsed, 4),
            "elapsed_min": round(elapsed / 60.0, 4),
        })

    round_total_sec = time.perf_counter() - round_t0

    round_summaries.append({
        "round": round_idx,
        "round_total_sec": round(round_total_sec, 4),
        "round_total_min": round(round_total_sec / 60.0, 4),
        "seg_sec": round(round_seg_sec, 4),
        "seg_min": round(round_seg_sec / 60.0, 4),
        "classify_sec": round(round_classify_sec, 4),
        "classify_min": round(round_classify_sec / 60.0, 4),
        "integrity_check_sec": round(round_check_sec, 4),
        "integrity_check_min": round(round_check_sec / 60.0, 4),
    })

overall_total_sec = time.perf_counter() - overall_t0
ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =========================
# 4) 輸出單一時間報表
# =========================
final_report_dir = CLASSIFY_OUTPUT_ROOT / train_model_method / Map_number
final_report_dir.mkdir(parents=True, exist_ok=True)

txt_path = final_report_dir / "time_report.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("SAM + Classifier Loop Time Report\n")
    f.write("=" * 100 + "\n")
    f.write(f"started_at        : {started_at}\n")
    f.write(f"ended_at          : {ended_at}\n")
    f.write(f"start_round       : {START_ROUND}\n")
    f.write(f"total_rounds      : {TOTAL_ROUNDS}\n")
    f.write(f"overall_total_sec : {overall_total_sec:.2f}\n")
    f.write(f"overall_total_min : {overall_total_sec / 60.0:.2f}\n")
    f.write("\n")
    f.write("Note: round_total_sec may be slightly larger than seg+classify+integrity_check\n")
    f.write("      because it includes controller overhead in the main script.\n")
    f.write("\n")

    f.write("[Round Summary]\n")
    f.write("-" * 100 + "\n")
    for row in round_summaries:
        f.write(f"Round {row['round']}\n")
        f.write(f"  round_total      : {row['round_total_sec']:.2f} sec ({row['round_total_min']:.2f} min)\n")
        f.write(f"  seg              : {row['seg_sec']:.2f} sec ({row['seg_min']:.2f} min)\n")
        f.write(f"  classify         : {row['classify_sec']:.2f} sec ({row['classify_min']:.2f} min)\n")
        f.write(f"  integrity_check  : {row['integrity_check_sec']:.2f} sec ({row['integrity_check_min']:.2f} min)\n")
        f.write("\n")

    f.write("[Module Detail]\n")
    f.write("-" * 100 + "\n")
    current_round = None
    for rec in time_records:
        if current_round != rec["round"]:
            current_round = rec["round"]
            f.write(f"Round {current_round}\n")
        f.write(
            f"  - {rec['module_name']:<30} "
            f"{rec['elapsed_sec']:.2f} sec ({rec['elapsed_min']:.2f} min)\n"
        )
    f.write("\n")

print("\nALL DONE")
print(f"[TIME REPORT] saved to: {txt_path}")