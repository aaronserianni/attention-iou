import argparse
import subprocess
import numpy as np
import os

parser = argparse.ArgumentParser(prog="run_subgroups", description="Run subgroup scripts")

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __str__(self):
        return f'[{self.start}, {self.end}]'
    
    def __repr__(self):
        return self.__str__()

parser.add_argument('target_label', type=str)
parser.add_argument('mcc', type=float,
                        choices=Range(-1.0, 1.0))
parser.add_argument('--data_dir', type=str,
                        dest="data_dir", required=True,
                        help="Dataset directory")

args = parser.parse_args()

low = np.round(max(args.mcc-.2, -.99), 2)
high = np.round(min(args.mcc+.2, .99), 2)

for i in np.round(np.arange(low, high+.001, .01), 2):
    # print([os.path.abspath('run_subgroups.slurm'), str(args.target_label), str(i), str(low), str(high)])
    print(f"python run_trainer.py --celeb --data_dir {str(args.data_dir)} --save_model models/ --subgroup_targets {str(args.target_label)} Male --subgroup_train_mcc {str(i)} --subgroup_train_mcc_range {str(low)} {str(high)} --train_multiple 10")
    print(f"python run_bias.py --celeb --data_dir {str(args.data_dir)} --results_dir results/subgroups/ --load_model models/ --score heatmap --heatmap_score_targets {str(args.target_label)} Male --subgroup_targets {str(args.target_label)} Male --subgroup_train_mcc {str(i)} --subgroup_train_mcc_range {str(low)} {str(high)} --score_sample --no-resize_attentions")
    print(f"python run_evaluations.py --celeb --data_dir {str(args.data_dir)} --results_dir results/subgroups/ --load_model models/ --subgroup_targets {str(args.target_label)} Male --subgroup_train_mcc {str(i)} --subgroup_train_mcc_range {str(low)} {str(high)} --score_sample --evaluate_model")
    # command = subprocess.Popen([os.path.abspath('run_subgroups.slurm'), str(args.target_label), str(i), str(low), str(high)])
