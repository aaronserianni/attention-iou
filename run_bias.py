import os
import time
import glob
import pickle

from bias import get_model
from waterbirds import WaterbirdsDataset, WaterbirdsScorer
from celeb import CelebDataset, CelebScorer
from coco import COCODataset, COCOScorer
from load_args import get_bias_args

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter


def main():
    args = get_bias_args()

    if args.waterbirds:
        print(f"Running bias scores on Waterbirds with bias percentage {args.waterbirds_percentage} with model {args.model}")
        n_classes = 2
    elif args.celeb:
        if args.score == 'confounder':
            print(f"Running bias confounder scores on CelebA with model target label {args.target_label} and confounder label {args.confounder_label}")
        if args.score == 'heatmap':
            print(f"Running bias heatmap scores on CelebA with model target label {args.target_label} and heatmap target labels {args.heatmap_score_targets}")
        if args.score == 'mask':
            print(f"Running bias mask scores on CelebA with model target label {args.target_label} and mask target labels {args.mask_score_target}")
        if len(args.target_label) == 1:
            args.target_label = args.target_label[0]
            n_classes = 40 if args.target_label == "all" else 2
        else:
            n_classes = len(args.target_label)     
    elif args.coco:
        if args.score == 'confounder':
            print(f"Running bias confounder scores on COCO with model target label {args.target_label} and confounder label {args.confounder_label}")
        if args.score == 'heatmap':
            print(f"Running bias heatmap scores on COCO with model target label {args.target_label}, heatmap target labels {args.heatmap_score_targets}")
        if args.score == 'mask':
            print(f"Running bias mask scores on COCO with model target label {args.target_label}, mask target labels {args.mask_score_target}")
        if len(args.target_label) == 1:
            args.target_label = args.target_label[0]
            n_classes = 171 if args.target_label == "all" else 2
        else:
            n_classes = len(args.target_label)
    print(f"Using model {args.model} {'normalized scores' if args.normalize_scores else ''}{', positive only scores' if args.positive_scores else ''}, and attribution method {args.attribution}")

    print("Loading data")
    if args.waterbirds:
        dataset = WaterbirdsDataset(
            os.path.join(os.path.abspath(args.data_dir), f"waterbird_complete{args.waterbirds_percentage}_forest2water2/"),
            args.resolution,
            augment_data=args.augment_data,
        )
    elif args.celeb:
        dataset = CelebDataset(
            os.path.abspath(args.data_dir),
            args.resolution,
            augment_data=args.augment_data,
            target_label=args.target_label,
            target_masks=args.target_masks,
            target_mcc=args.subgroup_train_mcc,
            target_mcc_range=args.subgroup_train_mcc_range,
            subgrouping_labels=args.subgroup_targets,
            one_sided_train=args.one_sided_train,
            one_sided_target=args.one_sided_target,
            flip_labels=args.flip_labels,
        )
    elif args.coco:
        dataset = COCODataset(
                os.path.abspath(args.data_dir),
                args.resolution,
                augment_data=args.augment_data,
                target_label=args.target_label,
                target_masks=args.target_masks,
            )
    print("Loaded data")

    (train_data, val_data, test_data) = dataset.prepare_dataloaders(
        train=True
    )

    val_loader = val_data.get_loader(
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = test_data.get_loader(
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    print("Created dataloaders")

    model = get_model(args.model, n_classes, False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, time.strftime("%c")))
    print("Tensorboard Dir: " + str(os.path.join(args.tensorboard_dir, time.strftime("%c"))))

    if args.waterbirds:
        scorer = WaterbirdsScorer(
            model,
            test_loader,
            device,
            args.attribution,
            writer,
            args.results_dir,
            args.normalize_scores,
            args.positive_scores,
            waterbirds_percentage=args.waterbirds_percentage,
            num_samples=10000,
            multiple_figures=args.multiple_figures,
            figure_frequency=args.figure_frequency,
            resize_attentions=args.resize_attentions
        )
    elif args.celeb:
        scorer = CelebScorer(
            model,
            test_loader if args.subgroup_train_mcc is None else val_loader,
            device,
            args.attribution,
            writer,
            args.results_dir,
            args.normalize_scores,
            args.positive_scores,
            args.target_label,
            args.target_masks,
            args.confounder_label,
            num_samples=10000,
            multiple_figures=args.multiple_figures,
            figure_frequency=args.figure_frequency,
            subgroup_mcc=args.subgroup_train_mcc,
            subgrouping_labels=args.subgroup_targets,
            one_sided_train=args.one_sided_train,
            one_sided_target=args.one_sided_target,
            resize_attentions=args.resize_attentions
        )
    elif args.coco:
        scorer = COCOScorer(
            model,
            test_loader,
            device,
            args.attribution,
            writer,
            args.results_dir,
            args.normalize_scores,
            args.positive_scores,
            args.target_label,
            args.target_masks,
            args.confounder_label,
            num_samples=10000,
            multiple_figures=args.multiple_figures,
            figure_frequency=args.figure_frequency,
            resize_attentions=args.resize_attentions,
            label_dir=args.data_dir,
        )
    
    if not args.score_sample:
        if os.path.basename(args.load_model) == '':
            load_path = scorer.get_model_path(scorer.multi_target, args)
        else:
            load_path = os.path.abspath(args.load_model)
        print(f"Loading model from {load_path}")
        model.load_state_dict(torch.load(load_path))

    if args.attribution == "shortlist":
        shortlist_torchray = [
            "DeConvNet",
            "ExcitationBackprop",
            "Gradient",
            "GuidedBackprop",
            "LinearApprox",
            "GradCAM",
        ]
        
        for i, attribution in enumerate(shortlist_torchray):
            if args.figure_frequency == 0:
                scorer.attribution_number = i
            scorer.attribution_method = attribution
            scorer.calculate_confounder_score()
    elif args.score_sample: # TODO: CLEAN THIS UP
        score_list = {k: [] for k in scorer.score_functions.keys()}
        raw_score_list = {k: [] for k in scorer.score_functions.keys()}
        correct_list = []
        label_list = []
        pred_list = []
        
        load_path = scorer.get_model_path(scorer.multi_target, args, multiple=True)

        for i, file in enumerate(glob.glob(load_path)):
            print(f"Loading model from {file}")
            model.load_state_dict(torch.load(file))

            if args.score == "confounder":
                scores, correct_array, label_array, pred_array = scorer.calculate_confounder_score(sample_scores=False)
            elif args.score == "heatmap":
                scores, correct_array, label_array, pred_array = scorer.calculate_heatmap_score(target1=args.heatmap_score_targets[0], target2=args.heatmap_score_targets[1], sample_scores=False)
            elif args.score == "mask":
                if args.celeb or args.coco:
                    scores, correct_array, label_array, pred_array = scorer.calculate_mask_score(scorer.MASK_GROUPS, args.mask_score_target, sample_scores=False)
                if args.waterbirds:
                    scores, correct_array, label_array, pred_array = scorer.calculate_mask_score(scorer.MASK_GROUPS, sample_scores=False)
            for key in scorer.score_functions.keys():
                score_list[key].append(np.nanmean(scores[key], axis=-1))
                raw_score_list[key].append(scores[key])
            correct_list.append(correct_array)
            label_list.append(label_array)
            pred_list.append(pred_array)


        for key in scorer.score_functions.keys():
            if args.score == "heatmap":
                scores = np.stack(score_list[key], axis=-1).squeeze(1)
            elif args.score == "mask":
                scores = np.stack(score_list[key], axis=-1)
            elif args.score == "confounder":
                scores = np.stack(score_list[key], axis=-1)
            all_scores = np.concatenate((np.nanmean(scores, axis=-1), np.nanstd(scores, axis=-1)), axis=0)
            if args.score == "confounder":
                scorer.save_scores(key, all_scores, is_averaged=True)
            elif args.score == "heatmap":
                scorer.save_scores(key, all_scores, is_averaged=True, target1=args.heatmap_score_targets[0], target2=args.heatmap_score_targets[1])
            elif args.score == "mask":
                scorer.save_scores(key, all_scores, is_averaged=True, is_mask_group=True, target=args.mask_score_target)

        if args.save_raw_scores:
            raw_scores_dict = {'scores': {}}
            for key in scorer.score_functions.keys():
                if args.score == "heatmap":
                    raw_scores_dict['scores'][key] = np.stack(raw_score_list[key], axis=-1).squeeze(1)
                    file_name = f'Heatmap-{"all" if len(scorer.target_label) == 171 else scorer.target_label}_{args.heatmap_score_targets[0]}_{args.heatmap_score_targets[1]}_{scorer.attribution_method}{"_normalized" if scorer.normalize_scores else ""}{"_positive" if scorer.positive_scores else ""}.pkl'
                elif args.score == "mask":
                    raw_scores_dict['scores'][key] = np.stack(raw_score_list[key], axis=-1)
                    file_name = f'Masks-{"all" if len(scorer.target_label) == 171 else scorer.target_label}_{args.mask_score_target}_{scorer.attribution_method}{"_normalized" if scorer.normalize_scores else ""}{"_positive" if scorer.positive_scores else ""}.pkl'
                elif args.score == "confounder":
                    raw_scores_dict['scores'][key] = np.stack(raw_score_list[key], axis=-1)
                    file_name = f'{scorer.target_label}_{scorer.target_masks}_{scorer.attribution_method}{"_normalized" if scorer.normalize_scores else ""}{"_positive" if scorer.positive_scores else ""}.pkl'
            raw_scores_dict['labels'] = np.stack(label_list, axis=-1)
            raw_scores_dict['preds'] = np.stack(pred_list, axis=-1)

            print(raw_scores_dict['scores']['element_wise'].shape)
            print(raw_scores_dict['labels'].shape)
            print(raw_scores_dict['preds'].shape)
            
            print(f"Saving raw scores to {os.path.join(args.save_raw_scores, file_name)}")
            with open(os.path.join(args.save_raw_scores, file_name), 'wb+') as handle:
                pickle.dump(raw_scores_dict, handle)

    else:
        if args.score == "confounder":
            scorer.calculate_confounder_score()
        elif args.score == "heatmap":
            scorer.calculate_heatmap_score(target1=args.heatmap_score_targets[0], target2=args.heatmap_score_targets[1])
        elif args.score == "mask":
            if args.celeb or args.coco:
                scorer.calculate_mask_score(scorer.MASK_GROUPS, args.mask_score_target, uniform_attention=args.uniform_attention)
            if args.waterbirds:
                scorer.calculate_mask_score(scorer.MASK_GROUPS)


if __name__ == '__main__':
    main()
