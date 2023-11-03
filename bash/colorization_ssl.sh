#!/bin/bash
# a bash script for depth completion through colorization

# change your parameters
sensor=livox_all # cepton, livox_all
bin_num=5 # 1,5,10,25
width=1250
height=375
data=val # train, val, test
use_gpu=True # True, False
winRad=1 # 1,2...
kernel=cross # cross, full

dataset=${sensor}'/'${sensor}'_'${bin_num}
input_path='./data/'${dataset}'/'${data}'/sparse_depth'
output_path='./results/ssl/'${width}'x'${height}'/'${data}'/'${dataset}
ground_truth_path='./data/'${dataset}'/'${data}'/ground_truth'
eval_output_path='./eval_results/ssl/'${width}'x'${height}'/'${data}'/'${dataset}

echo "generating results from colorization..."
python3 ./src/colorization.py \
--input_path $input_path \
--output_path $output_path \
--eval_output_path $eval_output_path \
--use_gpu $use_gpu \
--winRad $winRad \
--kernel $kernel \

echo "finished generating results for colorization method..."

echo "evaluating results..."
# normalise = 1 for normalise the metric, 0 for not normalise the metric
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
