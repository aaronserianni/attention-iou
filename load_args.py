import sys
import argparse


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


def get_trainer_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_trainer", description="Run training code")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--waterbirds', action='store_true', help="Train and run on Waterbirds dataset")
    group.add_argument('--celeb', action='store_true', help="Train and run on CelebA dataset")
    group.add_argument('--coco', action='store_true', help="Train and run on COCO dataset")
    parser.add_argument('--data_dir', type=str,
                        dest="data_dir", required=True,
                        help="Dataset directory")
    parser.add_argument('--tensorboard_dir', type=str,
                        dest="tensorboard_dir", default="tensorboard/", required=False,
                        help="Tensorboard directory")
    parser.add_argument('--device', type=str,
                        dest="device", default="cuda:0", required=False,
                        help="Device to use (i.e. cpu, cuda:0)")
    parser.add_argument('--resolution', type=int,
                        dest="resolution", default=224, required=False,
                        help="Input resolution for model")
    parser.add_argument('--batch_size', type=int,
                        dest="batch_size", default=64, required=False,
                        help="Batch size during training")
    parser.add_argument('--num_workers', type=int,
                        dest="num_workers", default=4, required=False,
                        help="Number of workers for DataLoader")
    parser.add_argument('--pin_memory',
                        dest="pin_memory", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Enable automatic memory pinning in data loading')
    parser.add_argument('--augment',
                        dest="augment_data", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Augment the training data')
    parser.add_argument('--num_epochs',
                        dest="num_epochs", type=int,
                        default=10, required=False,
                        help='Number of epochs to train for')
    parser.add_argument('--model', type=str,
                        default="resnet50_extended", dest="model",
                        help="Model to use (resnet18, resnet50, resnet50_extended, efficientnet, vit_b32)")
    parser.add_argument('--weights',
                        dest="weights", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Use pretrained weights')
    parser.add_argument('--multiple_figures',
                        dest="multiple_figures", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Save multiple images per figure to Tensorboard')
    parser.add_argument('--figure_frequency',
                        dest="figure_frequency", type=int,
                        default=0, required=False,
                        help='Frequency in batches for which to save figures to tensorboard (0 for only once)')
    parser.add_argument('--waterbirds_percentage', type=int,
                        dest="waterbirds_percentage",
                        required='--waterbirds' in sys.argv,
                        help="Waterbirds dataset percentage to train on")
    parser.add_argument('--target_label', type=str,
                        nargs='+',
                        dest="target_label",
                        default=["all"],
                        help="Target CelebA/COCO attribute to train on, can inclue multiple")
    parser.add_argument('--save_model', type=str,
                        dest="save_model",
                        required=True,
                        help="Save model to specified folder as a state_dict")
    parser.add_argument('--train_multiple', type=int,
                        dest="train_multiple", required=False,
                        help="Train and save model multiple times")
    parser.add_argument('--train_index', type=int,
                        dest="train_index", required=False,
                        help="Train and save model for specified number")
    parser.add_argument('--target_masks', type=str,
                        nargs='+',
                        dest="target_masks",
                        default=["skin"] if '--celeb' in sys.argv else (["Person"] if '--coco' in sys.argv else None),
                        required='confounder' in sys.argv and ('--celeb' in sys.argv or '--coco' in sys.argv),
                        help="Target CelebA/COCO masks for confounder score to evaluate on")
    parser.add_argument('--subgroup_train_mcc', type=float,
                        dest="subgroup_train_mcc", 
                        required=False,
                        choices=Range(-1.0, 1.0),
                        help="Target MCC of subgroups in train set, and make Celeb dataset with these training subgroups")
    parser.add_argument('--subgroup_train_mcc_range', type=float,
                        dest="subgroup_train_mcc_range",
                        nargs=2,
                        required='subgroup_train_mcc' in sys.argv,
                        choices=Range(-1.0, 1.0),
                        help="Target MCC of subgroups in train set, and make Celeb dataset with these training subgroups")
    parser.add_argument('--subgroup_targets', type=str,
                        nargs=2, metavar=('target1', 'target2'),
                        dest="subgroup_targets",
                        required='subgroup_train_mcc' in sys.argv,
                        help="Target CelebA attributes to run subgroups on")
    parser.add_argument('--one_sided_train', type=str,
                        choices=['positive', 'negative'],
                        dest="one_sided_train",
                        required=False,
                        help="Train on only positive or negative labels in dataset")
    parser.add_argument('--one_sided_target', type=str,
                        dest="one_sided_target", 
                        required='one_sided_train' in sys.argv and '--celeb' in sys.argv,
                        help="Target CelebA attribute to do one-sided training on")
    parser.add_argument('--flip_labels', action=argparse.BooleanOptionalAction,
                        dest="flip_labels", 
                        default=False, required=False,
                        help="Flip labels of Celeb dataset")
    
    args = parser.parse_args()

    return args


def get_bias_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_bias", description="Run bias code")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--waterbirds', action='store_true', help="Train and run on Waterbirds dataset")
    group.add_argument('--celeb', action='store_true', help="Train and run on CelebA dataset")
    group.add_argument('--coco', action='store_true', help="Train and run on COCO dataset")
    parser.add_argument('--data_dir', type=str,
                        dest="data_dir", required=True,
                        help="Dataset directory")
    parser.add_argument('--tensorboard_dir', type=str,
                        dest="tensorboard_dir", default="tensorboard/", required=False,
                        help="Tensorboard directory")
    parser.add_argument('--results_dir', type=str,
                        dest="results_dir", default="results/", required=False,
                        help="Results directory")
    parser.add_argument('--device', type=str,
                        dest="device", default="cuda:0", required=False,
                        help="Device to use (i.e. cpu, cuda:0)")
    parser.add_argument('--resolution', type=int,
                        dest="resolution", default=224, required=False,
                        help="Input resolution for model")
    parser.add_argument('--batch_size', type=int,
                        dest="batch_size", default=64, required=False,
                        help="Batch size during training")
    parser.add_argument('--num_workers', type=int,
                        dest="num_workers", default=4, required=False,
                        help="Number of workers for DataLoader")
    parser.add_argument('--pin_memory',
                        dest="pin_memory", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Enable automatic memory pinning in data loading')
    parser.add_argument('--augment',
                        dest="augment_data", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Augment the training data')
    parser.add_argument('--model', type=str,
                        default="resnet50_extended", dest="model",
                        help="Model to use (resnet18, resnet50, resnet50_extended, efficientnet, vit_b32)")
    parser.add_argument('--multiple_figures',
                        dest="multiple_figures", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Save multiple images per figure to Tensorboard')
    parser.add_argument('--figure_frequency',
                        dest="figure_frequency", type=int,
                        default=0, required=False,
                        help='Frequency in batches for which to save figures to tensorboard (0 for only once)')
    parser.add_argument('--attribution',
                        dest="attribution", type=str,
                        default="GradCAM", required=False,
                        help="Attribution method to use (GradCAM, ScoreCAM, EigenCAM, EigenGradCAM, or all)")
    parser.add_argument('--normalize_scores',
                        dest="normalize_scores", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Normalize bias scores')
    parser.add_argument('--positive_scores',
                        dest="positive_scores", action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Positive only bias scores')
    parser.add_argument('--waterbirds_percentage', type=int,
                        dest="waterbirds_percentage",
                        required='--waterbirds' in sys.argv,
                        help="Waterbirds dataset percentage to train on")
    parser.add_argument('--target_label', type=str,
                        nargs='+',
                        dest="target_label",
                        default=["all"],
                        help="Target CelebA/COCO attribute to train on, can inclue multiple")
    parser.add_argument('--confounder_label', type=str,
                        dest="confounder_label",
                        default="Male" if '--celeb' in sys.argv else ("Person" if '--coco' in sys.argv else None),
                        help="Confounding CelebA/COCO attribute to evaluate on")
    parser.add_argument('--load_model', type=str,
                        dest="load_model", required=True,
                        help="Load model from specified state_dict")
    parser.add_argument('--score_sample', action=argparse.BooleanOptionalAction,
                        default=False,
                        dest="score_sample", required=False,
                        help="Get sample of scores based on pretrained models in folder")
    parser.add_argument('--average_maps',
                        dest="average_maps", action=argparse.BooleanOptionalAction,
                        default=False, required=False,
                        help='Calculate and display average attention map and average mask')
    parser.add_argument('--average_maps_groups', type=str,
                        nargs=2, metavar=('target1', 'target2'),
                        dest="average_maps_groups",
                        required=False,
                        help='Calculate and display average average mask')
    parser.add_argument('--average_masks',
                        dest="average_masks", action=argparse.BooleanOptionalAction,
                        default=False, required=False,
                        help='Calculate and display average attention map and average mask')
    parser.add_argument('--score',
                        dest="score", type=str,
                        choices=['confounder', 'heatmap', 'mask'],
                        required=True,
                        help='Choose which score type to run')
    parser.add_argument('--target_masks', type=str,
                        nargs='+',
                        dest="target_masks",
                        default=["skin"] if '--celeb' in sys.argv else (["Person"] if '--coco' in sys.argv else None),
                        required='confounder' in sys.argv and ('--celeb' in sys.argv or '--coco' in sys.argv),
                        help="Target CelebA/COCO masks for confounder score to evaluate on")
    parser.add_argument('--heatmap_score_targets', type=str,
                        nargs=2, metavar=('target1', 'target2'),
                        dest="heatmap_score_targets",
                        required='heatmap' in sys.argv and ('--celeb' in sys.argv or '--coco' in sys.argv),
                        help="Target CelebA/COCO attributes to calculate heatmap score")
    parser.add_argument('--mask_score_target', type=str,
                        dest="mask_score_target", 
                        required='mask' in sys.argv and ('--celeb' in sys.argv or '--coco' in sys.argv),
                        help="Target CelebA/COCO attribute to calculate mask score")
    parser.add_argument('--subgroup_train_mcc', type=float,
                        dest="subgroup_train_mcc", 
                        required=False,
                        choices=Range(-1.0, 1.0),
                        help="Target MCC of subgroups in train set, and make Celeb dataset with these training subgroups")
    parser.add_argument('--subgroup_train_mcc_range', type=float,
                        dest="subgroup_train_mcc_range",
                        nargs=2,
                        required='subgroup_train_mcc' in sys.argv,
                        choices=Range(-1.0, 1.0),
                        help="Target MCC of subgroups in train set, and make Celeb dataset with these training subgroups")
    parser.add_argument('--subgroup_targets', type=str,
                        nargs=2, metavar=('target1', 'target2'),
                        dest="subgroup_targets",
                        required='subgroup_train_mcc' in sys.argv,
                        help="Target CelebA attributes to run subgroups on")
    parser.add_argument('--one_sided_train', type=str,
                        choices=['positive', 'negative'],
                        dest="one_sided_train",
                        required=False,
                        help="Train on only positive or negative labels in dataset")
    parser.add_argument('--one_sided_target', type=str,
                        dest="one_sided_target", 
                        required='one_sided_train' in sys.argv and '--celeb' in sys.argv,
                        help="Target CelebA attribute to do one-sided training on")
    parser.add_argument('--resize_attentions',
                        dest="resize_attentions", action=argparse.BooleanOptionalAction,
                        default=False if 'mask' in sys.argv and ('--celeb' in sys.argv or '--coco' in sys.argv) else True, required=False,
                        help='Resize attention maps')
    parser.add_argument('--flip_labels', action=argparse.BooleanOptionalAction,
                        dest="flip_labels", 
                        default=False, required=False,
                        help="Flip labels of Celeb dataset")
    parser.add_argument('--uniform_attention', action=argparse.BooleanOptionalAction,
                        dest="uniform_attention", 
                        default=False, required=False,
                        help="Run mask score with uniform attention")
    parser.add_argument('--save_raw_scores', type=str,
                        dest="save_raw_scores", required=False,
                        help="Save raw scores, labels, and predictions as pickled dictionary for COCO")

    args = parser.parse_args()

    return args


def get_evaluation_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_evaluations", description="Run evaluation code")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--waterbirds', action='store_true', help="Train and run on Waterbirds dataset")
    group.add_argument('--celeb', action='store_true', help="Train and run on CelebA dataset")
    group.add_argument('--coco', action='store_true', help="Train and run on COCO dataset")
    parser.add_argument('--data_dir', type=str,
                        dest="data_dir", required=True,
                        help="Dataset directory")
    parser.add_argument('--tensorboard_dir', type=str,
                        dest="tensorboard_dir", default="tensorboard/", required=False,
                        help="Tensorboard directory")
    parser.add_argument('--results_dir', type=str,
                        dest="results_dir", default="results/", required=False,
                        help="Results directory")
    parser.add_argument('--device', type=str,
                        dest="device", default="cuda:0", required=False,
                        help="Device to use (i.e. cpu, cuda:0)")
    parser.add_argument('--resolution', type=int,
                        dest="resolution", default=224, required=False,
                        help="Input resolution for model")
    parser.add_argument('--batch_size', type=int,
                        dest="batch_size", default=64, required=False,
                        help="Batch size during training")
    parser.add_argument('--num_workers', type=int,
                        dest="num_workers", default=4, required=False,
                        help="Number of workers for DataLoader")
    parser.add_argument('--pin_memory',
                        dest="pin_memory", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Enable automatic memory pinning in data loading')
    parser.add_argument('--augment',
                        dest="augment_data", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Augment the training data')
    parser.add_argument('--model', type=str,
                        default="resnet50_extended", dest="model",
                        help="Model to use (resnet18, resnet50, resnet50_extended, efficientnet, vit_b32)")
    parser.add_argument('--multiple_figures',
                        dest="multiple_figures", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Save multiple images per figure to Tensorboard')
    parser.add_argument('--figure_frequency',
                        dest="figure_frequency", type=int,
                        default=0, required=False,
                        help='Frequency in batches for which to save figures to tensorboard (0 for only once)')
    parser.add_argument('--attribution',
                        dest="attribution", type=str,
                        default="GradCAM", required=False,
                        help="Attribution method to use (GradCAM, ScoreCAM, EigenCAM, EigenGradCAM, or all)")
    parser.add_argument('--normalize_scores',
                        dest="normalize_scores", action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Normalize bias scores')
    parser.add_argument('--positive_scores',
                        dest="positive_scores", action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Positive only bias scores')
    parser.add_argument('--waterbirds_percentage', type=int,
                        dest="waterbirds_percentage",
                        required='--waterbirds' in sys.argv,
                        help="Waterbirds dataset percentage to train on")
    parser.add_argument('--target_label', type=str,
                        nargs='+',
                        dest="target_label",
                        default=["all"],
                        help="Target CelebA/COCO attribute to train on, can inclue multiple")
    parser.add_argument('--confounder_label', type=str,
                        dest="confounder_label",
                        default="Male" if '--celeb' in sys.argv else ("Person" if '--coco' in sys.argv else None),
                        help="Confounding CelebA/COCO attribute to evaluate on")
    parser.add_argument('--load_model', type=str,
                        dest="load_model", required=True,
                        help="Load model from specified state_dict")
    parser.add_argument('--score_sample', action=argparse.BooleanOptionalAction,
                        default=False,
                        dest="score_sample", required=False,
                        help="Get sample of scores based on pretrained models in folder")
    parser.add_argument('--average_maps',
                        dest="average_maps", action=argparse.BooleanOptionalAction,
                        default=False, required=False,
                        help='Calculate and display average attention map')
    parser.add_argument('--average_maps_groups', type=str,
                        nargs=2, metavar=('target1', 'target2'),
                        dest="average_maps_groups",
                        required=False,
                        help='Calculate and display average maps for confounding labels')
    parser.add_argument('--average_masks',
                        dest="average_masks", action=argparse.BooleanOptionalAction,
                        default=False, required=False,
                        help='Calculate and display average mask')
    parser.add_argument('--target_masks', type=str,
                        nargs='+',
                        dest="target_masks",
                        default=["skin"] if '--celeb' in sys.argv else (["Person"] if '--coco' in sys.argv else None),
                        required='confounder' in sys.argv and ('--celeb' in sys.argv or '--coco' in sys.argv),
                        help="Target CelebA/COCO masks for confounder score to evaluate on")
    parser.add_argument('--subgroup_train_mcc', type=float,
                        dest="subgroup_train_mcc", 
                        required=False,
                        choices=Range(-1.0, 1.0),
                        help="Target MCC of subgroups in train set, and make Celeb dataset with these training subgroups")
    parser.add_argument('--subgroup_train_mcc_range', type=float,
                        dest="subgroup_train_mcc_range",
                        nargs=2,
                        required='subgroup_train_mcc' in sys.argv,
                        choices=Range(-1.0, 1.0),
                        help="Target MCC of subgroups in train set, and make Celeb dataset with these training subgroups")
    parser.add_argument('--subgroup_targets', type=str,
                        nargs=2, metavar=('target1', 'target2'),
                        dest="subgroup_targets",
                        required='subgroup_train_mcc' in sys.argv,
                        help="Target CelebA attributes to run subgroups on")
    parser.add_argument('--one_sided_train', type=str,
                        choices=['positive', 'negative'],
                        dest="one_sided_train",
                        required=False,
                        help="Train on only positive or negative labels in dataset")
    parser.add_argument('--one_sided_target', type=str,
                        dest="one_sided_target", 
                        required='one_sided_train' in sys.argv and '--celeb' in sys.argv,
                        help="Target CelebA attribute to do one-sided training on")
    parser.add_argument('--evaluate_model',
                        dest="evaluate_model", action=argparse.BooleanOptionalAction,
                        default=False, required=False,
                        help='Evaluate and save model results')
    parser.add_argument('--resize_attentions',
                        dest="resize_attentions", action=argparse.BooleanOptionalAction,
                        default=False if 'mask' in sys.argv and ('--celeb' in sys.argv or '--coco' in sys.argv) else True, required=False,
                        help='Resize attention maps')
    parser.add_argument('--directional_bias_amplification', type=str,
                        dest="directional_bias_amplification", 
                        required=False,
                        help="Run directional bias amplifcation on specified attribute")
    parser.add_argument('--flip_labels', action=argparse.BooleanOptionalAction,
                        dest="flip_labels", 
                        default=False, required=False,
                        help="Flip labels of Celeb dataset")
    parser.add_argument('--test_accuracy', action=argparse.BooleanOptionalAction,
                        dest="test_accuracy", 
                        default=False, required=False,
                        help="Calculate accuracy and AP")

    args = parser.parse_args()

    return args