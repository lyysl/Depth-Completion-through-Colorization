#!/bin/bash
# a bash script to reconstruct images from colorization method
# bash ./bash/colorization_kitti.sh

# change your parameters
data=val # validation dataset is tested
use_gpu=True # True, False
winRad=1 # 1,2...
kernel=cross # cross, full

input_path='./data/data_depth_selection/depth_selection/val_selection_cropped/velodyne_raw'
ground_truth_path='./data/data_depth_selection/depth_selection/val_selection_cropped/groundtruth_depth'
output_path='./results/kitti/'${data}'/'${kernel}
eval_output_path='./eval_results/kitti/'${data}'/'${kernel}

echo "generating results for colorization method..."
python3 ./src/colorization_kitti.py \
--input_path $input_path \
--output_path $output_path \
--eval_output_path $eval_output_path \
--use_gpu $use_gpu \
--winRad $winRad \
--kernel $kernel \

echo "finished generating results for colorization method..."

echo "evaluating results..."
python src/evaluation.py \
--output_depth_path $output_path \
--ground_truth_path $ground_truth_path \
--n_batch 8 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--save_outputs \
--output_path $eval_output_path \
--normalise 0 \

echo "finish evaluating results..."
