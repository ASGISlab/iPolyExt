import ray
ray.init('local')
import os, sys
import logging
import argparse
import numpy as np
import pandas as pd
from time import time
from rich.progress import track

import cmaas_utils.io as io
from cmaas_utils.types import MapUnitType
from cmaas_utils.logging import start_logger, changeConsoleHandler
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from submodules.validation.src.grading import grade_point_raster, grade_poly_raster, usgs_grade_poly_raster
from rich.logging import RichHandler
#copy from submodules/validation/validationDemo.py
LOGGER_NAME = 'DARPA_CMAAS_VALIDATION'
FILE_LOG_LEVEL = logging.DEBUG
STREAM_LOG_LEVEL = logging.INFO

def parse_command_line():
    """Runs Command line argument parser for pipeline. Exit program on bad arguments. Returns struct of arguments"""
    from typing import List
    def parse_data(path: str) -> List[str]:
        """Command line argument parser for --data. --data should accept a list of file and/or directory paths as an
           input. This function is called on each individual element of that list and checks if the path is valid."""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist'
            raise argparse.ArgumentTypeError(msg+'\n')
        return path
    
    def post_parse_data(data : List[str]) -> List[str]:
        """Loops over all data arguments and finds all tif files. If the path is a directory expands it to all the valid
           files paths inside the dir. Returns a list of valid files. Raises an argument exception if no valid files were given"""
        data_files = []
        for path in data:
            # Check if its a directory
            if os.path.isdir(path):
                data_files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')])
            if os.path.isfile(path) and path.endswith('.tif'):
                data_files.append(path)
        if len(data_files) == 0:
            msg = 'No valid files where given to --data argument. --data should be given a path or paths to file(s) \
                    and/or directory(s) containing the data to perform inference on. program will only run on .tif files'
            raise argparse.ArgumentTypeError(msg)
        return data_files
    
    def parse_directory(path : str) -> str:
        """Command line argument parser for directory path arguments. Raises argument error if the path does not exist
           or if it is not a valid directory. Returns directory path"""
        # Check if it exists
        if not os.path.exists(path):
            msg = f'Invalid path "{path}" specified : Path does not exist\n'
            raise argparse.ArgumentTypeError(msg)
        # Check if its a directory
        if not os.path.isdir(path):
            msg = f'Invalid path "{path}" specified : Path is not a directory\n'
            raise argparse.ArgumentTypeError(msg)
        return path

    def parse_feature(feature_string : str) -> str:
        """Command line argument parser for --feature_type. Case insensitive, accepts point, pt, polygon or poly.
           Raises an argument exception if argument is one of these list. Returns lowercase feature type"""
        # Convert string to type
        feature = MapUnitType.from_str(feature_string)
        # Check if feature type is supported
        if feature not in [MapUnitType.POINT, MapUnitType.POLYGON]:
            msg = f'Invalid feature type "{feature_string}" specified.\nAvailable feature types are :\n\t* Point\n\t* Polygon'
            raise argparse.ArgumentTypeError(msg)
        return feature
    
    parser = argparse.ArgumentParser(description='', add_help=False)
    required_args = parser.add_argument_group('required arguments', '')
    required_args.add_argument('-p', '--pred_segmentations', 
                        type=parse_directory,
                        required=True,
                        help='Path to file(s) and/or directory(s) containing the predicted rasters to grade. The \
                              program will run grade on any .tif files.')            
    required_args.add_argument('-t','--true_segmentations',
                        type=parse_data,
                        required=True,
                        nargs='+',
                        help='Directory containing the true raster segmentations to score the data against.')
    required_args.add_argument('-m','--map_images',
                        type=parse_directory,
                        required=True,
                        help='Directory containing the base map images for the segmentations')
    required_args.add_argument('-l', '--legends',
                        type=parse_directory,
                        required=True,
                        help='Directory containing precomputed legend data in USGS json format.')

    # Optional Arguments
    optional_args = parser.add_argument_group('optional arguments', '')
    optional_args.add_argument('-o', '--output',
                        default='results',
                        help='Directory to write the validation feedback to. Defaults to "results"')
    optional_args.add_argument('--log',
                        default='logs/Latest.log',
                        help='Option to set the file logging will output to. Defaults to "logs/Latest.log"')
    optional_args.add_argument('--min_valid_range',
                        type=float,
                        default=0.1,
                        help='Maximum distance in % of the largest size of the image (diagonal) ' +
                             'between a predicted pixel vs. a true one that will be considered ' +
                             'as valid to include in the scoring. Defaults to 0.1')
    optional_args.add_argument('--difficult_weight',
                        type=float,
                        default=0.7,
                        help='Weight to give difficult points in the F1 score, range is a float within [0, 1]. Defaults to 0.7')
    optional_args.add_argument('--set_false_as',
                        default='hard',
                        help='when set to "hard" the pixels that are not within the true polygon area will be considered hard.'
                             'Set how to treat false positives and false negatives. Options are "hard" or "easy". Defaults to "hard"')
    optional_args.add_argument('--color_range',
                        type=int,
                        default=4,
                        help='The range of color variation to consider for the legend color. Defaults to 4')
    
    # Flags
    flag_group = parser.add_argument_group('Flags', '')
    flag_group.add_argument('-h', '--help',
                        action='help', 
                        help='show this message and exit')
    flag_group.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Flag to change the logging level from INFO to DEBUG')
    flag_group.add_argument('--feedback',
                        action='store_true',
                        help='Enable saving debugging feedback images.')

    args = parser.parse_args()
    args.true_segmentations = post_parse_data(args.true_segmentations)
    return args

@ray.remote
def inner_loop(true_segmentations, args, pbar):
    results_df = pd.DataFrame(columns = [
        'Map','Feature', 'F1 Score', 'Precision', 'Recall', 'IoU Score', 'Map Area',
        'USGS F1 Score', 'USGS Precision', 'USGS Recall', 
        'Mean matched distance Points', 'Matched Points', 'Unmatched Points', 'Missing Points'])

    print(f'Starting grading of {len(true_segmentations)} files')
    last_map_filepath = None
    last_legend_filepath = None
    skipped_files = []
    potential_map_names = [os.path.splitext(m)[0] for m in os.listdir(args.map_images) if m.endswith('.tif')]
    for gold_filepath in true_segmentations:
        pbar.update.remote(1)
        feature_name = os.path.basename(os.path.splitext(gold_filepath)[0])
        feature_type = MapUnitType.from_str(feature_name.split('_')[-1])
        print(f'Processing {feature_name}')
        map_name = [m for m in potential_map_names if m in gold_filepath][0]
        # Load data
        true_image, _, _ = io.loadGeoTiff(gold_filepath)
        try:
            pred_image, _, _ = io.loadGeoTiff(os.path.join(args.pred_segmentations, os.path.basename(gold_filepath)))
        except:
            print(f'No pred segementation map present for {feature_name}. Skipping')
            result = {
                'Map': map_name,
                'Feature': feature_name,
                'F1 Score' : 0, 
                'Precision' : 0, 
                'Recall' : 0,
                'Mean Matched Distance' : np.nan, 
                'Matched Points' : 0,
                'Missing Points' : np.count_nonzero(true_image), 
                'Map Area': np.prod(true_image.shape),
                'Unmatched Points' : 0
            }
            results_df.loc[len(results_df)] = result
            skipped_files.append(os.path.join(args.pred_segmentations, os.path.basename(gold_filepath)))
            continue

        
        map_filepath = os.path.join(args.map_images, map_name + '.tif')
        if map_filepath != last_map_filepath:
            map_image, crs, transform = io.loadGeoTiff(map_filepath)
            last_map_filepath = map_filepath
        legend_filepath = os.path.join(args.legends, map_name + '.json')
        if legend_filepath != last_legend_filepath:
            map_legend = io.loadLegendJson(legend_filepath)
            last_legend_filepath = legend_filepath
        feature = [f for f in map_legend.features if f.label.replace(' ', '_') in gold_filepath][0]

        feedback_image = None
        if args.feedback:
            feedback_image = np.zeros(map_image.shape, dtype=np.uint8)
            

        if feature_type == MapUnitType.POINT:
            results, feedback_image = grade_point_raster(pred_image, true_image, feedback_image=feedback_image, min_valid_range=args.min_valid_range)
            results['Map'] = map_name
            results['Feature'] = feature_name
            results['USGS F1 Score'] = results['F1 Score']
            results['USGS Precision'] = results['Precision']
            results['USGS Recall'] = results['Recall']

            print(f'Results for "{feature_name}" : ' + 
                f'F1 Score : {results["F1 Score"]:.2f}, ' + 
                f'Precision : {results["Precision"]:.2f}, ' +
                f'Recall : {results["Recall"]:.2f}, ' +
                f'Mean matched distance : {results["Mean Matched Distance"]:.2f}, ' +
                f'Matched pts : {results["Matched Points"]:.2f}, ' +
                f'Missing pts : {results["Missing Points"]:.2f}, ' +
                f'Unmatched pts : {results["Unmatched Points"]:.2f}'
            )
            results_df.loc[len(results_df)] = results
            
        if feature_type == MapUnitType.POLYGON:
            results, feedback_image = grade_poly_raster(pred_image, true_image, feedback_image=feedback_image, map_name=map_name)
            USGS_results, _ = usgs_grade_poly_raster(pred_image, true_image, map_image, feature.label_bbox, feedback_image=feedback_image, difficult_weight=args.difficult_weight, color_range=args.color_range, set_false_as=args.set_false_as)
            results['Map'] = map_name
            results['Feature'] = feature_name
            results['USGS F1 Score'] = USGS_results['F1 Score']
            results['USGS Precision'] = USGS_results['Precision']
            results['USGS Recall'] = USGS_results['Recall']

            print(f'Results for "{feature_name}" : ' + 
                f'F1 Score : {results["F1 Score"]:.2f}, ' + 
                f'Precision : {results["Precision"]:.2f}, ' +
                f'Recall : {results["Recall"]:.2f}, ' +
                f'IoU Score : {results["IoU Score"]:.2f}, ' +
                f'USGS F1 Score: {results["USGS F1 Score"]:.2f}, ' +
                f'USGS Precision : {results["USGS Precision"]:.2f}, ' +
                f'USGS Recall : {results["USGS Recall"]:.2f}'
            )    
            results_df.loc[len(results_df)] = results

        if feedback_image is not None:
            print(f'Saving feedback image for {feature_name}')
            io.saveGeoTiff(os.path.join(args.output, 'val_' + feature_name + '.tif'), feedback_image, crs, transform)
    return results_df, skipped_files

def main(args):
    main_time = time()
    # Start logger
    if args.verbose:
        global FILE_LOG_LEVEL, STREAM_LOG_LEVEL
        FILE_LOG_LEVEL = logging.DEBUG
        STREAM_LOG_LEVEL = logging.DEBUG
    global log
    log = start_logger(LOGGER_NAME, args.log, log_level=FILE_LOG_LEVEL, console_log_level=STREAM_LOG_LEVEL)

    # Log info statement to console even if in warning only mode
    log.handlers[1].setLevel(logging.INFO)
    log.info(f'Running pipeline on {os.uname()[1]} with following parameters:\n' +
            f'\tPred Data    : {args.pred_segmentations}\n' +
            f'\tTrue Data    : {args.true_segmentations}\n' +
            f'\tMaps         : {args.map_images}\n' +
            f'\tLegends      : {args.legends}\n' +
            f'\tOutput       : {args.output}\n')
    log.handlers[1].setLevel(STREAM_LOG_LEVEL)

    # Create Output Directory
    if not os.path.exists(args.output) and not os.path.splitext(args.output)[1]:
        os.makedirs(args.output)
    from ray.experimental import tqdm_ray
    remote_tqdm = ray.remote(tqdm_ray.tqdm)
    pbar = remote_tqdm.remote(total=len(args.true_segmentations))
    mid_list_parts = [list(item) for item in np.array_split(args.true_segmentations, 16)]
    future = [
        inner_loop.remote(chunk, args, pbar) 
        for i, chunk in enumerate(mid_list_parts) if len(chunk) > 0
    ]
    results_df = pd.DataFrame(columns = [
        'Map','Feature', 'F1 Score', 'Precision', 'Recall', 'IoU Score', 'Map Area',
        'USGS F1 Score', 'USGS Precision', 'USGS Recall', 
        'Mean matched distance Points', 'Matched Points', 'Unmatched Points', 'Missing Points'])
    skipped_files = []
    for results_df_part, skipped_files_part in ray.get(future):
        results_df = pd.concat([results_df, results_df_part], ignore_index=True)
        skipped_files.extend(skipped_files_part)

    csv_path = os.path.join(args.output, 'validationDemo_results.csv')
    results_df.fillna(0).to_csv(csv_path)
    if len(skipped_files) > 0:
        log.warning(f'Skipped grading on {len(skipped_files)} files. Could not find the necessary files for the following: {skipped_files}')
    log.info(f'Finished grading, saving results to {csv_path}, Runtime was {time()-main_time} seconds')    
    

if __name__=='__main__':
    args = parse_command_line()
    main(args)