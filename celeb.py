import functools
import math
from operator import itemgetter
import os
from typing import Union
import warnings

import pandas as pd
import torch
import numpy as np
from scipy import optimize
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import DataLoader

from bias import BiasBase
from bias_dataset import BiasDataset
from bias_trainer import BiasTrainer
from bias_scorer import BiasScorer
from ap_utils import normalized_ap_wrapper

warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

def mcc(groups):
    groups = np.reshape(groups, (2, 2))
    return (groups[1, 1] * groups[0, 0] - groups[1, 0] * groups[0, 1]) / math.sqrt(
                groups[0].sum()
                * groups[1].sum()
                * groups[:, 0].sum()
                * groups[:, 1].sum()
            )

class CelebBase(BiasBase):

    MASK_LIST = [
        "background",
        "skin",
        "nose",
        "eye_g",
        "l_eye",
        "r_eye",
        "l_brow",
        "r_brow",
        "l_ear",
        "r_ear",
        "mouth",
        "u_lip",
        "l_lip",
        "hair",
        "hat",
        "ear_r",
        "neck_l",
        "neck",
        "cloth",
    ]
    MASK_GROUPS = {
        "background": ["background"],
        "skin": ["skin"],
        "nose": ["nose"],
        "eyeglasses": ["eye_g"],
        "eyes": ["l_eye", "r_eye"],
        "eyebrows": ["l_brow", "r_brow"],
        "ears": ["l_ear", "r_ear"],
        "mouth": ["mouth", "u_lip", "l_lip"],
        "hair": ["hair"],
        "hat": ["hat"],
        "earrings": ["ear_r"],
        "necklace": ["neck_l"],
        "neck": ["neck"],
        "cloth": ["cloth"],
    }
    LABEL_DICT = {
        "5_o_Clock_Shadow": 0,
        "Arched_Eyebrows": 1,
        "Attractive": 2,
        "Bags_Under_Eyes": 3,
        "Bald": 4,
        "Bangs": 5,
        "Big_Lips": 6,
        "Big_Nose": 7,
        "Black_Hair": 8,
        "Blond_Hair": 9,  # works well
        "Blurry": 10,
        "Brown_Hair": 11,
        "Bushy_Eyebrows": 12,
        "Chubby": 13,
        "Double_Chin": 14,
        "Eyeglasses": 15,  # very easy to detect
        "Goatee": 16,
        "Gray_Hair": 17,
        "Heavy_Makeup": 18,
        "High_Cheekbones": 19,
        "Male": 20,
        "Mouth_Slightly_Open": 21,  # works well
        "Mustache": 22,
        "Narrow_Eyes": 23,
        "No_Beard": 24,
        "Oval_Face": 25,
        "Pale_Skin": 26,
        "Pointy_Nose": 27,
        "Receding_Hairline": 28,
        "Rosy_Cheeks": 29,
        "Sideburns": 30,
        "Smiling": 31,
        "Straight_Hair": 32,
        "Wavy_Hair": 33,  # inconsitencies in labeling wavy vs straight
        "Wearing_Earrings": 34,
        "Wearing_Hat": 35,
        "Wearing_Lipstick": 36,
        "Wearing_Necklace": 37,
        "Wearing_Necktie": 38,
        "Young": 39,
    }
    SHORT_LABEL_LIST = [
        "Arched_Eyebrows",
        "Bangs",
        "Blond_Hair",
        "High_Cheekbones",
        "Male",
        "Mouth_Slightly_Open",
        "Pointy_Nose",
        "Smiling",
        "Wearing_Earrings",
        "Wearing_Lipstick",
    ]
    HARD_LABEL_LIST = [
        "Arched_Eyebrows",
        "Bags_Under_Eyes",
        "Big_Nose",
        "Black_Hair",
        "Bushy_Eyebrows",
        "Male",
        "Pointy_Nose",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Lipstick",
    ]

    LABEL_LIST_DICT = {"SHORT_LABEL_LIST": SHORT_LABEL_LIST, "HARD_LABEL_LIST": HARD_LABEL_LIST}

    def __init__(self, target_label: Union[str, list[str]], target_masks: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if target_label == "all":
            self.target_label = list(self.LABEL_DICT.keys())
        else:
            self.target_label = target_label
        self.target_masks = [self.MASK_LIST.index(target_mask) for target_mask in target_masks]
        self.multi_target = not isinstance(self.target_label, str)

    def get_masks(self, masks, mask_group=None):
        if mask_group:
            target_masks = [self.MASK_LIST.index(target_mask) for target_mask in mask_group]
            return (sum(torch.round(masks * 255) == target_mask for target_mask in target_masks)).int()
        else:
            return (sum(torch.round(masks * 255) == target_mask for target_mask in self.target_masks)).int()

    def get_target_labels(self, labels, target_label=None):
        if target_label:
            return labels[:, self.LABEL_DICT[target_label]]
        elif self.multi_target:
            return labels[:, itemgetter(*self.target_label)(self.LABEL_DICT)]
        else:
            return labels[:, self.LABEL_DICT[self.target_label]]

    def get_target_value(self, target_name):
        if self.multi_target:
            return self.target_label.index(target_name)
        else:
            return 1

    def get_figure_title(self, labels):
        if self.multi_target:
            if len(self.target_label) == 40:
                return ["Image {0}, Label all".format(i) for i in range(16)]
            else:
                return ["Image {0}, Label {1}".format(i, self.target_label) for i in range(16)]
                # return ["Image {0}, Label {1}: {2}".format(i, self.target_label, self.get_target_labels(labels)[i].bool().tolist()) for i in range(16)]
        else:
            return ["Image {0}, Label {1}: {2}".format(i, self.target_label, bool(self.get_target_labels(labels)[i])) for i in range(16)]
        
    def get_model_path(self, multi_target, args, index=None, multiple=False):
        folder_path = args.save_model if hasattr(args, 'save_mode') and args.save_model else args.load_model
        if multi_target and args.target_label != "all":
            load_path = os.path.join(os.path.abspath(folder_path), f'celeb_{"-".join(args.target_label)}{"_mcc" + str(args.subgroup_train_mcc + "-".join(args.subgroup_targets)) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}_{args.model}{"_" + str(index) if index is not None else ""}{"_*" if multiple else ""}.pt')
        else:
            load_path = os.path.join(os.path.abspath(folder_path), f'celeb_{args.target_label}{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}_{args.model}{"_" + str(index) if index is not None else ""}{"_*" if multiple else ""}.pt')
        
        return load_path


class CelebDataset(CelebBase, BiasDataset):

    def __init__(
        self,
        data_dir: str,
        target_resolution: tuple[int, int],
        target_label: Union[str, list[str]],
        target_masks: list[str],
        augment_data: bool = False,
        target_mcc: float = None,
        target_mcc_range: list[float] = None,
        subgrouping_labels: list[str] = None,
        train_split: int = 0.7,
        val_split: int = 0.15,
        test_split: int = 0.15,
        seed: int = 0,
        flip_labels: bool = False,
        one_sided_train: bool = None,
        one_sided_target: str = None
    ):
        super().__init__(
            data_dir=data_dir,
            target_resolution=target_resolution,
            augment_data=augment_data,
            target_label=target_label,
            target_masks=target_masks,
        )

        self.image_dir = os.path.join(data_dir, "CelebA-HQ-img")
        self.mask_dir = os.path.join(data_dir, "CelebAMask-HQ-mask")
        self.label_df = pd.read_csv(
            os.path.join(data_dir, "CelebAMask-HQ-attribute-anno.csv"), sep="\\s+"
        )
        self.label_df.replace({-1: 0}, inplace=True)

        assert train_split + val_split + test_split == 1

        num_images = len(self.label_df)

        np.random.seed(seed)
        torch.manual_seed(seed)
        indices = np.random.permutation(num_images)
        self.train_idx = indices[: int(num_images * train_split)]
        self.val_idx = indices[int(num_images * train_split) : int(num_images * (train_split + val_split))]
        self.test_idx = indices[int(num_images * (1 - test_split)) :]

        split_array = np.zeros(num_images)
        split_array[self.train_idx] = 0
        split_array[self.val_idx] = 1
        split_array[self.test_idx] = 2

        self.split_array = split_array

        self.filename_array = self.label_df['filename'].values
        self.label_array = self.label_df.loc[:, self.label_df.columns != 'filename'].astype(int).to_numpy()
        if flip_labels:
            self.label_array = 1 - self.label_array

        self.target_mcc = target_mcc
        self.target_mcc_range = target_mcc_range
        self.subgrouping_labels = subgrouping_labels
        self.set_size = len(self.train_idx)
        if self.subgrouping_labels is not None:
            assert len(self.subgrouping_labels) == 2
            assert len(self.target_mcc_range) == 2

            self.set_size = np.min(np.concatenate([np.sum(self.get_subgroups_smooth(self.target_mcc_range[0], self.train_idx)[2], axis=1), 
                                              np.sum(self.get_subgroups_smooth(self.target_mcc_range[1], self.train_idx)[2], axis=1)]))
            print(f"Subsampling {self.set_size} of {len(self.train_idx)} images for training set with subgrouping.")

            subgroups = self.get_subgroups_smooth(self.target_mcc, self.train_idx, total=self.set_size)[0]
            self.split_array = self.regroup_splits(split_array, subgroups)
        
        self.one_sided_train = one_sided_train
        self.one_sided_target = one_sided_target
        if self.one_sided_train is not None:
            assert self.one_sided_target in self.LABEL_DICT
            if one_sided_train == "positive":
                mask = (
                    (self.label_array[:, self.LABEL_DICT[self.one_sided_target]] == 0)
                    & (split_array == 0)
                )
            elif one_sided_train == "negative":
                mask = (
                    (self.label_array[:, self.LABEL_DICT[self.one_sided_target]] == 1)
                    & (split_array == 0)
                )
            split_array[np.where(mask)[0]] = 4
            print(f"Using {np.unique(split_array, return_counts=True)[1][0]} of {len(self.train_idx)} images with {self.one_sided_train} {self.one_sided_target} labels for training set.")

        self.loss_weights = self.get_loss_weights(self.label_array[self.train_idx])
        self.train_class_weights = self.get_class_weights(self.label_array[self.train_idx])

    def __len__(self):
        return len(self.label_df)

    def get_img_name(self, idx: int):
        return os.path.join(self.image_dir, self.filename_array[idx])

    def get_mask_name(self, idx: int):
        return os.path.join(
            self.mask_dir, self.filename_array[idx].split(".")[0] + ".png"
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        return super().__getitem__(idx)

    def get_loss_weights(self, label_array):
        if self.multi_target:
            balance = np.mean(label_array, axis=0)
            weights = (1 - balance) / balance
            weights = weights[list(itemgetter(*self.target_label)(self.LABEL_DICT))]
        else:
            weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=self.get_target_labels(label_array))
        return weights

    def get_class_weights(self, label_array):
        if self.multi_target:
            class_weight_dict = []
            for target in self.target_label:
                class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=self.get_target_labels(label_array, target))
                class_weight_dict.append({0: class_weights[0], 1: class_weights[1]})
        else:
            class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=self.get_target_labels(label_array))
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        return class_weight_dict
    
    def get_subgroups_smooth(self, target_mcc, indices, total=None):
        starting_groups = self.label_df.iloc[indices].groupby(by=self.subgrouping_labels).size().unstack(fill_value=0).stack()
        starting_mcc = mcc(starting_groups)
        if starting_mcc < target_mcc:
            list_mccs = -np.round(np.arange(-target_mcc, -starting_mcc, 0.01), 3)
        else:
            list_mccs = np.round(np.arange(target_mcc, starting_mcc, 0.01), 3)
        if list_mccs[0] == target_mcc:
            list_mccs = list_mccs[::-1]
        subgroups = []
        for list_mcc in list_mccs:
            subgroups.append(self.get_subgroups(list_mcc, indices, starting_groups=subgroups[-1] if len(subgroups) > 0 else starting_groups, total=total))
        return subgroups[-1], list_mccs, subgroups


    def get_subgroups(self, target_mcc, indices, starting_groups=None, print_results=False, total=None):
        def least_squares(groups, original_groups):
            return ((groups - original_groups) ** 2 / (groups**2 + 1e-15)).sum()
        
        input_groups = self.label_df.iloc[indices].groupby(by=self.subgrouping_labels).size().unstack(fill_value=0).stack()
        if starting_groups is None:
            starting_groups = input_groups

        if print_results:
            print(f"Original groups {np.reshape(input_groups, 4)} with MCC {mcc(input_groups)}")
        bounds = optimize.Bounds(0, np.reshape(input_groups, 4))
        constraint = optimize.NonlinearConstraint(mcc, target_mcc, target_mcc)
        if total is not None:
            total_constraint = optimize.LinearConstraint([1, 1, 1, 1], total, total)
        min = optimize.minimize(
            functools.partial(least_squares, original_groups=starting_groups),
            np.reshape(input_groups, 4),
            constraints=[constraint] if total is None else [constraint, total_constraint],
            method='trust-constr',
            tol=1e-6,
            bounds=bounds,
            options={"maxiter": 1000000},
        )
        if min.success is False:
            raise RuntimeError(f"Minimizer failed to reach target MCC, due to error '{min.message}'")
        subgroups = np.rint(min.x).astype(int)
        if print_results:
            print(f"Found subgroups of size {subgroups} with MCC {mcc(subgroups)}")
        return subgroups

    def regroup_splits(self, split_array, subgroups):
        for i in [0, 1]:
            for j in [0, 1]:
                mask = (
                    (self.label_array[:, self.LABEL_DICT[self.subgrouping_labels[0]]] == i)
                    & (self.label_array[:, self.LABEL_DICT[self.subgrouping_labels[1]]] == j)
                    & (split_array == 0)
                )
                indices = np.random.permutation(sum(mask))
                split_array[np.where(mask)[0][indices[subgroups[2 * i + j] :]]] = 4
        train_mcc = mcc(
            self.label_df.iloc[np.where(split_array == 0)[0]]
            .groupby(by=self.subgrouping_labels)
            .size().unstack(fill_value=0).stack()
        )
        val_mcc = mcc(
            self.label_df.iloc[np.where(split_array == 1)[0]]
            .groupby(by=self.subgrouping_labels)
            .size().unstack(fill_value=0).stack()
        )
        test_mcc = mcc(
            self.label_df.iloc[np.where(split_array == 2)[0]]
            .groupby(by=self.subgrouping_labels)
            .size().unstack(fill_value=0).stack()
        )
        print(f"Subgroup train MCC: {train_mcc}, Val MCC: {val_mcc}, Test MCC: {test_mcc}")
        return split_array


class CelebTrainer(CelebBase, BiasTrainer):

    def __init__(
        self,
        model,
        dataset: BiasDataset,
        device,
        optimizer,
        scheduler,
        loss_function,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        writer,
        target_label: Union[str, list[str]],
        target_masks: list[str],
        multiple_figures=True,
        figure_frequency=0,
    ) -> None:
        super().__init__(
            model=model,
            dataset=dataset,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            writer=writer,
            target_label=target_label,
            target_masks=target_masks,
            multiple_figures=multiple_figures,
            figure_frequency=figure_frequency
        )

    def per_class_accuracy(self, pred_array, label_array, print_results=False, confounder = "Male"):
        average_precision = normalized_ap_wrapper(label_array, pred_array)
        average_precision2 = average_precision_score(label_array, pred_array, average=None)
        accuracy = np.zeros(len(self.target_label))
        group_accuracies = np.zeros((len(self.target_label), 4))
        group_sizes = np.zeros((len(self.target_label), 4))
        for i, target in enumerate(self.target_label):
            sample_weights = compute_sample_weight(class_weight=self.dataset.train_class_weights[i], y=label_array[:, i])
            # average_precision[i] = average_precision_score(label_array, pred_array, sample_weight=sample_weights, average=None)[i]
            accuracy[i] = accuracy_score(label_array[:, i], np.round(pred_array[:, i]), sample_weight=sample_weights)
            accuracy2 = accuracy_score(label_array[:, i], np.round(pred_array[:, i]))
            true_true_mask = np.logical_and(self.get_target_labels(label_array, target) == 1, self.get_target_labels(label_array, confounder) == 1)
            true_false_mask = np.logical_and(self.get_target_labels(label_array, target) == 1, self.get_target_labels(label_array, confounder) == 0)
            false_true_mask = np.logical_and(self.get_target_labels(label_array, target) == 0, self.get_target_labels(label_array, confounder) == 1)
            false_false_mask = np.logical_and(self.get_target_labels(label_array, target) == 0, self.get_target_labels(label_array, confounder) == 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                group_accuracies[i][0] = accuracy_score(label_array[true_true_mask, i], np.round(pred_array[true_true_mask, i]))
                group_accuracies[i][1] = accuracy_score(label_array[true_false_mask, i], np.round(pred_array[true_false_mask, i]))
                group_accuracies[i][2] = accuracy_score(label_array[false_true_mask, i], np.round(pred_array[false_true_mask, i]))
                group_accuracies[i][3] = accuracy_score(label_array[false_false_mask, i], np.round(pred_array[false_false_mask, i]))
            group_sizes[i][0] = np.sum(true_true_mask)
            group_sizes[i][1] = np.sum(true_false_mask)
            group_sizes[i][2] = np.sum(false_true_mask)
            group_sizes[i][3] = np.sum(false_false_mask)
            if print_results is True:
                print(f"{target} Test Result: Positive Labels {np.sum(label_array[:, i])} | Accuracy {accuracy[i]:.4f} | Accuracy Unweighted {accuracy2:.4f} | Average Precision: {average_precision[i]:.4f} | Average Precision Unweighted: {average_precision2[i]:.4f}")
        return average_precision, accuracy, group_accuracies, group_sizes

class CelebScorer(CelebBase, BiasScorer):

    def __init__(
        self,
        model,
        test_loader,
        device,
        attribution_method,
        writer,
        save_directory,
        normalize_scores,
        positive_scores,
        target_label,
        target_masks,
        confounder_label,
        num_samples=10000,
        multiple_figures=True,
        figure_frequency=0,
        subgroup_mcc=None,
        subgrouping_labels=None,
        one_sided_train=None,
        one_sided_target=None,
        resize_attentions=True
    ):
        super().__init__(
            model=model,
            test_loader=test_loader,
            device=device,
            attribution_method=attribution_method,
            writer=writer,
            save_directory=save_directory,
            normalize_scores=normalize_scores,
            positive_scores=positive_scores,
            num_samples=num_samples,
            target_label=target_label,
            target_masks=target_masks,
            multiple_figures=multiple_figures,
            figure_frequency=figure_frequency,
            resize_attentions=resize_attentions
        )
        self.confounder_label = confounder_label
        self.subgroup_mcc = subgroup_mcc
        self.subgrouping_labels = subgrouping_labels
        self.one_sided_train = one_sided_train
        self.one_sided_target = one_sided_target

    def get_confounder_labels(self, labels, confounder_label=None):
        if confounder_label is None:
            return labels[:, self.LABEL_DICT[self.confounder_label]]
        return labels[:, self.LABEL_DICT[confounder_label]]

    def score_table(self, scores, labels, target1=None, target2=None, target=None, preds=None):
        true_true_scores = []
        true_false_scores = []
        false_true_scores = []
        false_false_scores = []

        if target:
            true_true_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels, target), self.get_confounder_labels(labels)), scores, torch.nan))
            true_false_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels, target), torch.logical_not(self.get_confounder_labels(labels))), scores, torch.nan))
            false_true_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels, target)), self.get_confounder_labels(labels)), scores, torch.nan))
            false_false_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels, target)), torch.logical_not(self.get_confounder_labels(labels))), scores, torch.nan))
            tile_length = 1
        elif self.multi_target and target1 and target2:
            true_true_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels, target1), self.get_target_labels(labels, target2)), scores, torch.nan))
            true_false_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels, target1), torch.logical_not(self.get_target_labels(labels, target2))), scores, torch.nan))
            false_true_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels, target1)), self.get_target_labels(labels, target2)), scores, torch.nan))
            false_false_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels, target1)), torch.logical_not(self.get_target_labels(labels, target2))), scores, torch.nan))
            tile_length = 1
        elif self.confounder_label in self.LABEL_LIST_DICT:
            for confounder_label in self.LABEL_LIST_DICT[self.confounder_label]:
                true_true_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels), self.get_confounder_labels(labels, confounder_label)), scores, torch.nan))
                true_false_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels), torch.logical_not(self.get_confounder_labels(labels, confounder_label))), scores, torch.nan))
                false_true_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels)), self.get_confounder_labels(labels, confounder_label)), scores, torch.nan))
                false_false_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels)), torch.logical_not(self.get_confounder_labels(labels, confounder_label))), scores, torch.nan))
            tile_length = len(self.LABEL_LIST_DICT[self.confounder_label])
        else:
            true_true_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels), self.get_confounder_labels(labels)), scores, torch.nan))
            true_false_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels), torch.logical_not(self.get_confounder_labels(labels))), scores, torch.nan))
            false_true_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels)), self.get_confounder_labels(labels)), scores, torch.nan))
            false_false_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels)), torch.logical_not(self.get_confounder_labels(labels))), scores, torch.nan))
            tile_length = 1
        return torch.stack(
            (
                torch.vstack(true_true_scores),
                torch.vstack(true_false_scores),
                torch.vstack(false_true_scores),
                torch.vstack(false_false_scores),
                torch.tile(scores, (tile_length, 1)),
            ), axis=0)

    def get_average_map_labeled(self, maps, labels):
        target_true = torch.where(self.get_target_labels(labels).bool().unsqueeze(1).unsqueeze(1), maps, torch.nan).nanmean(axis=0, keepdim=True)
        target_false = torch.where(torch.logical_not(self.get_target_labels(labels)).bool().unsqueeze(1).unsqueeze(1), maps, torch.nan).nanmean(axis=0, keepdim=True)
        maps_list = [(f"Target {self.target_label} true", target_true), (f"Target {self.target_label} false", target_false)]

        if self.confounder_label in self.LABEL_LIST_DICT:
            for confounder_label in self.LABEL_LIST_DICT[self.confounder_label]:
                confounder_true = torch.where(self.get_confounder_labels(labels, confounder_label).bool().unsqueeze(1).unsqueeze(1), maps, torch.nan).nanmean(axis=0, keepdim=True)
                confounder_false = torch.where(torch.logical_not(self.get_confounder_labels(labels, confounder_label)).bool().unsqueeze(1).unsqueeze(1), maps, torch.nan).nanmean(axis=0, keepdim=True)
                maps_list.extend([(f"Confounder {confounder_label} true", confounder_true), (f"Confounder {confounder_label} false", confounder_false)])
        else:
            confounder_true = torch.where(self.get_confounder_labels(labels).bool().unsqueeze(1).unsqueeze(1), maps, torch.nan).nanmean(axis=0, keepdim=True)
            confounder_false = torch.where(torch.logical_not(self.get_confounder_labels(labels)).bool().unsqueeze(1).unsqueeze(1), maps, torch.nan).nanmean(axis=0, keepdim=True)
            maps_list.extend([(f"Confounder {self.confounder_label} true", confounder_true), (f"Confounder {self.confounder_label} false", confounder_false)])

        return maps_list

    def save_scores(
        self,
        score_name,
        all_scores,
        label_array=None,
        correct_array=None,
        target1=None,
        target2=None,
        target=None,
        is_averaged=False,
        is_mask_group=False
    ):
        os.makedirs(os.path.dirname(self.save_directory), exist_ok=True)
        if len(all_scores.shape) == 1:
            all_scores = np.transpose(all_scores) 
        if is_mask_group:
            np.savetxt(os.path.join(os.path.dirname(self.save_directory), f'Masks-{"all" if len(self.target_label) == 40 else self.target_label}{"_mcc" + str(self.subgroup_mcc) + str(self.subgrouping_labels) if self.subgroup_mcc is not None else ""}{"_one-sided-" + self.one_sided_train + "-" + self.one_sided_target if self.one_sided_train is not None else ""}_{target}_{self.confounder_label}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv'), all_scores, delimiter=',', fmt='%.8f')
        elif self.multi_target and target1 and target2:
            np.savetxt(os.path.join(os.path.dirname(self.save_directory), f'Heatmap-{"all" if len(self.target_label) == 40 else self.target_label}{"_mcc" + str(self.subgroup_mcc) + str(self.subgrouping_labels) if self.subgroup_mcc is not None else ""}{"_one-sided-" + self.one_sided_train + "-" + self.one_sided_target if self.one_sided_train is not None else ""}_{target1}_{target2}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv'), all_scores, delimiter=',', fmt='%.8f')
        else:
            target_masks = [self.MASK_LIST[target_mask] for target_mask in self.target_masks]
            np.savetxt(os.path.join(os.path.dirname(self.save_directory), f'{self.target_label}_{target_masks}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv'), all_scores, delimiter=',', fmt='%.8f')
        if not self.multi_target and label_array is not None and correct_array is not None:
            if self.confounder_label in self.LABEL_LIST_DICT:
                print("Counfounder label true accuracy")
                for label in self.LABEL_LIST_DICT[self.confounder_label]:
                    print(np.nanmean(np.where(self.get_confounder_labels(label_array, label), correct_array, np.nan)))
                print("Counfounder label false accuracy")
                for label in self.LABEL_LIST_DICT[self.confounder_label]:
                    print(np.nanmean(np.where(np.logical_not(self.get_confounder_labels(label_array, label)), correct_array, np.nan)))
            else:
                print("Counfounder label true accuracy")
                print(np.nanmean(np.where(self.get_confounder_labels(label_array), correct_array, np.nan)))
                print("Counfounder label false accuracy")
                print(np.nanmean(np.where(np.logical_not(self.get_confounder_labels(label_array)), correct_array, np.nan)))

    def get_average_map_title(self, subset: str, index=None):
        if index is None:
            return f"Target: {self.target_label}, Subset: {subset}"
        else:
            return f"Target: {'all' if len(self.target_label) == 40 else self.target_label} / {self.target_label[index]}, Subset: {subset}"
