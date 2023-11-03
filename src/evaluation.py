# python scripts to evaluate the metric
# https://github.com/alexklwong/learning-topology-synthetic-data

'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Safa Cicek <safacicek@ucla.edu>

If this code is useful to you, please cite the following paper:
A. Wong, S. Cicek, and S. Soatto. Learning topology from synthetic data for unsupervised depth completion.
In the Robotics and Automation Letters (RA-L) 2021 and Proceedings of International Conference on Robotics and Automation (ICRA) 2021

@article{wong2021learning,
    title={Learning topology from synthetic data for unsupervised depth completion},
    author={Wong, Alex and Cicek, Safa and Soatto, Stefano},
    journal={IEEE Robotics and Automation Letters},
    volume={6},
    number={2},
    pages={1495--1502},
    year={2021},
    publisher={IEEE}
}
'''
import os, time, argparse
import numpy as np
import data_utils, eval_utils
from log_utils import log

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]


parser = argparse.ArgumentParser()

# result path
parser.add_argument('--output_depth_path',
    type=str, required=True, help='output depth path')
# Input paths
parser.add_argument('--ground_truth_path',
    type=str, default='', help='Paths to ground truth paths')
# Dataloader settings
parser.add_argument('--start_idx',
    type=int, default=0, help='Start of subset of samples to run')
parser.add_argument('--end_idx',
    type=int, default=1000, help='End of subset of samples to run')
# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=8, help='Number of samples per batch')
# Depth evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.0, help='Minimum depth value evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum depth value to evaluate')
# Output options
parser.add_argument('--save_outputs',
    action='store_true', help='If set, then save outputs')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep original input filenames')
parser.add_argument('--output_path',
    type=str, default='output', help='Path to save outputs')
parser.add_argument('--normalise',
    type=int, default=0, help='normalise the metric')

args = parser.parse_args()


'''
Read output depth paths and load ground truth (if available)
'''
log_path = os.path.join(args.output_path, 'results.txt')

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

# Load output depth from file for evaluation
output_depth_paths = get_imlist(args.output_depth_path)
output_depth_paths = output_depth_paths[args.start_idx:args.end_idx]

n_sample = len(output_depth_paths)

# Pad all paths based on batch size
output_depth_paths = data_utils.pad_batch(output_depth_paths, args.n_batch)

n_step = n_sample // args.n_batch

ground_truth_available = True if args.ground_truth_path != '' else False
ground_truths = []

if ground_truth_available:
    ground_truth_paths = get_imlist(args.ground_truth_path)
    ground_truth_paths = ground_truth_paths[args.start_idx:args.end_idx]

    n_sample = len(ground_truth_paths)

    # Load ground truth
    for idx in range(n_sample):

        print('Loading {}/{} groundtruth depth maps'.format(idx + 1, n_sample), end='\r')

        ground_truth, validity_map = \
            data_utils.load_depth_with_validity_map(ground_truth_paths[idx])

        ground_truth = np.concatenate([
            np.expand_dims(ground_truth, axis=-1),
            np.expand_dims(validity_map, axis=-1)], axis=-1)
        ground_truths.append(ground_truth)
    ground_truths = np.array(ground_truths)

    print('Completed loading {} groundtruth depth maps'.format(n_sample))


'''
evaluation
'''

output_depths = []
# Load the results from different methods
for idx in range(n_sample):

    print('Loading {}/{} output depth maps'.format(idx + 1, n_sample), end='\r')
    output_depth = data_utils.load_depth(output_depth_paths[idx])
    output_depths.append(output_depth)
output_depths = np.array(output_depths)


if ground_truth_available:
    # Run evaluation metrics
    eval_utils.evaluate(
        output_depths,
        ground_truths,
        log_path=log_path,
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        normalise=args.normalise)

