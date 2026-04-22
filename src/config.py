import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='my_exp', help="name of the run that will be used to name the output folder of various dirs.")
    parser.add_argument("--map_dir", type=str, default="data/map", help="path to your map directory, which contains the original maps along with the layout json")
    parser.add_argument("--contour_map_dir", type=str, default="data/contour_map", help="path to your map directory, which contains the original maps along with the layout json")
    parser.add_argument("--legend_dir", type=str, default="data/Legend", help="path to the directory containing the legend json files")
    parser.add_argument("--legend_patch_dir", type=str, default="data/Legend", help="path to the directory containing the legends patch image")
    parser.add_argument("--map_out_dir", type=str, default="data/cropped_map", help="path of the 5 augmented Geological Maps of Taiwan during the Japanese Colonial Period")
    parser.add_argument("--map", type=str, default=None, help="name of the map to process")
    parser.add_argument("--model_dir", type=str, default='model', help="path to the directory containing the trained classifier checkpoints and the sam model")
    parser.add_argument("--eval_key_dir", type=str, default="data/evaluation_answer_key", help="path to the dir containing evaluation answer key for polygon extraction")
    parser.add_argument("--cls_key_dir", type=str, default="data/cls_answer_key", help="path to the dir containing evaluation answer key for legend classification")
    parser.add_argument('--out_dir', type=str, default='run_seg_cls_loop_result', help='directory to save the output of sam & cls results.')
    parser.add_argument("--preproc_mode", type=str, default="none", choices=["none", "global_consistent", "paper_tint"], help="preprocessing mode: none = no preprocessing; global_consistent = global whole-image enhancement using fixed alpha/beta/gamma; paper_tint = paper-color tint / color-shift effect")
    parser.add_argument("--white_mask_area_target",type=float,default=0.21,help=(
            "Target white-mask area ratio for legend white occlusion augmentation. "
            "This controls noise_area_target_single.target."),)
    parser.add_argument("--white_mask_area_min",type=float,default=0.07,help=(
            "Minimum white-mask area ratio for legend white occlusion augmentation. "
            "This controls noise_area_target_single.min."
        ),)
    parser.add_argument("--white_mask_area_max",type=float,default=0.36,help=(
            "Maximum white-mask area ratio for legend white occlusion augmentation. "
            "This controls noise_area_target_single.max."
        ),)
    parser.add_argument("--white_mask_cand_tries_num",type=int,default=6,help=(
            "Number of candidate masks sampled for each image in legend white occlusion augmentation. "
            "Larger values make the generated white-mask area closer to the target but slower."
        ),)
    parser.add_argument("--cls_dir", type=str, default="data/train_data", help="path to the dir containing training data for classification")
    parser.add_argument("--cls_method", type=str, default="gen_legend_ratio_augmentation", help="data augmentation method, gen_legend_ratio_augmentation for our proposed method or GeoNet_Orig_augmentation or the baseline", choices=["gen_legend_ratio_augmentation", "GeoNet_Orig_augmentation"])
    parser.add_argument("--cls_ckpt", type=str, default=None, help="name of the classifier checkpoint folder")
    parser.add_argument("--cls_data", type=str, default=None, help="name of the classifier data folder")
    parser.add_argument("--cls_eval_target", type=str, default='legends', choices=['legends', 'others'], help="use classifier to evaluate lengends or others classes")
    parser.add_argument(
        "--run_compare_table",
        action="store_true",
        help=(
            "If set, run the optional GeoNet-vs-Ours comparison table after "
            "generating the current classifier evaluation summary. "
            "By default, only the current method summary is generated."
        ),
    )
    parser.add_argument("--cls_bs", type=int, default=64, help="number of patches in a batch to infer the legend type, adjust this accroding to your gpu.")
    parser.add_argument("--cls_procs", type=int, default=4, help="specify the number of classifier process that will be created on each gpu. too many cls process may cause OOM since we will load images onto memory.")
    parser.add_argument("--sam_bs", type=int, default=16, help="specify the points_per_batch of sam model, adjust this according to your gpu.")
    parser.add_argument("--sam_procs", type=int, default=4, help="specify the number of sam process that will be created on each gpu.")
    parser.add_argument("--num_preload", type=int, default=32, help="number of imtermediate images to preload into memory for the classifier process, adjust this according to your RAM and map size. too many preloaded image may cause OOM")
    parser.add_argument("--gpu_ids", type=str, default='0,1,2,3', help="which gpu will be used when creating sam/cls process, for example '0,1' if you have two gpus, or '0' if you have one gpu.")
    parser.add_argument("--num_workers", type=int, default=16, help="number of worker threads for parallel processing")
    parser.add_argument("--start_round", type=int, default=1, choices=[1, 2, 3, 4], help="which round to start running from")
    parser.add_argument("--underscore_replace",type=str, default='-', help="replace underscore in map name and legend name with this string, since our pipeline will use underscore as delimiter to parse the legend name and map name.")
    parser.add_argument(
        "--sam_pred_iou_thresh_by_round",
        type=str,
        default="0.85,0.75,0.60,0.60",
        help="SAM pred_iou_thresh for round 1-4, comma-separated."
    )
    parser.add_argument(
        "--sam_stability_score_thresh_by_round",
        type=str,
        default="0.85,0.75,0.60,0.60",
        help="SAM stability_score_thresh for round 1-4, comma-separated."
    )
    parser.add_argument(
        "--vote_majority_th_by_round",
        type=str,
        default="1.0,0.90,0.80,0.65",
        help="classifier majority threshold for round 1-4, comma-separated."
    )
    parser.add_argument(
        "--vote_min_patch_by_round",
        type=str,
        default="5,10,5,3",
        help="minimum patch count for classification vote in round 1-4, comma-separated."
    )
    
    parser.add_argument(
        "--vote_min_nonblank_by_round",
        type=str,
        default="2,2,2,2",
        help="minimum non-blank patch count for classification vote in round 1-4, comma-separated."
    )
    parser.add_argument(
        "--vote_strong_min_by_round",
        type=str,
        default="1,1,1,1",
        help="minimum strong vote count for classification vote in round 1-4, comma-separated."
    )
    parser.add_argument(
        "--blank_white_area_ratio",
        type=float,
        default=0.5,
        help=(
            "White-area ratio threshold for treating an uncovered residual mask as small. "
            "0.5 is equivalent to half of one classifier patch area."
        ),
    )
    
    parser.add_argument(
        "--small_mask_extra_width",
        type=int,
        default=60,
        help=(
            "Extra width used to compute the small residual mask threshold: "
            "PATCH_H * PATCH_W + PATCH_H * small_mask_extra_width."
        ),
    )
    #with the default setting each gpu will use around 26GB of RAM when running SAM
    return parser.parse_args()

# This executes once when the module is first imported
args = _parse_args()