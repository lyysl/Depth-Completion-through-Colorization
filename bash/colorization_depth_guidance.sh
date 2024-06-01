#!/bin/bash
# a bash script to reconstruct images from colorization method with guidance
# bash ./bash/colorization_depth_guidance.sh

# change your parameters
data=val # validation dataset is tested
use_gpu=True # True, False
winRad=1 # only neighbor of radius 1 is used for guidance
kernel=cross # only cross is used for guidance

input_path='./data/kitti/data_depth_selection/depth_selection/val_selection_cropped/velodyne_raw'
intrinsics_path='./data/kitti/data_depth_selection/depth_selection/val_selection_cropped/intrinsics'
guidance_path='./results/kitti/'${data}'/'${kernel}
output_path='./results/kitti_guidance/'${data}'/'${kernel}
ground_truth_path='./data/kitti/data_depth_selection/depth_selection/val_selection_cropped/groundtruth_depth'
eval_output_path='./eval_results/kitti_guidance/'${data}'/'${kernel}

echo "generating results for colorization method..."
python3 ./src/colorization_depth_guidance.py \
--input_path $input_path \
--intrinsics_path $intrinsics_path \
--guidance_path $guidance_path \
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
