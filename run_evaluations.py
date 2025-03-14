import os
import time
import glob
import re
import warnings
import pickle

from bias import biasamp_attribute_to_task, biasamp_task_to_attribute, get_model, BCEWithLogitsLossFlipped
from waterbirds import WaterbirdsDataset, WaterbirdsTrainer, WaterbirdsScorer
from celeb import CelebDataset, CelebTrainer, CelebScorer
from coco import COCODataset, COCOTrainer, COCOScorer
from load_args import get_evaluation_args

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as transforms
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


def main():
    args = get_evaluation_args()

    if args.waterbirds:
        n_classes = 2
    elif args.celeb:
        if len(args.target_label) == 1:
            args.target_label = args.target_label[0]
            n_classes = 40 if args.target_label == "all" else 2
        else:
            n_classes = len(args.target_label)     
    elif args.coco:
        if len(args.target_label) == 1:
            args.target_label = args.target_label[0]
            n_classes = 171 if args.target_label == "all" else 2
        else:
            n_classes = len(args.target_label)
    print(f"Using model {args.model}")

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

    train_loader = train_data.get_loader(
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
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

    if args.model == "vit_b32":
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.000001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=50)

    if dataset.multi_target:
        loss_function = BCEWithLogitsLossFlipped(pos_weight=torch.from_numpy(dataset.loss_weights).to(device), flipped=args.flip_labels)
    else:
        loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(dataset.loss_weights).float().to(device))

    if args.waterbirds:
        trainer = WaterbirdsTrainer(
            model,
            dataset,
            device,
            optimizer,
            scheduler,
            loss_function,
            train_loader,
            val_loader,
            test_loader,
            writer,
            multiple_figures=args.multiple_figures,
            figure_frequency=args.figure_frequency
        )
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
        trainer = CelebTrainer(
            model,
            dataset,
            device,
            optimizer,
            scheduler,
            loss_function,
            train_loader,
            val_loader,
            test_loader if args.subgroup_train_mcc is None else val_loader,
            writer,
            args.target_label,
            args.target_masks,
            multiple_figures=args.multiple_figures,
            figure_frequency=args.figure_frequency
        )
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
        trainer = COCOTrainer(
            model,
            dataset,
            device,
            optimizer,
            scheduler,
            loss_function,
            train_loader,
            val_loader,
            val_loader,
            writer,
            args.target_label,
            args.target_masks,
            multiple_figures=args.multiple_figures,
            figure_frequency=args.figure_frequency,
            label_dir=args.data_dir,
        )
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
    
    if args.average_masks:
        print(f"Running average masks, saving to {os.path.join(os.path.abspath(args.results_dir), 'average_masks/')}")
        for i, mask_group in enumerate(scorer.MASK_GROUPS):
            mask_array, label_array = scorer.get_all_masks(scorer.MASK_GROUPS[mask_group])
            if not args.resize_attentions:
                mask_array = transforms.functional.resize(
                    mask_array.float(),
                    size=(7, 7),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )

            figure = scorer.calculate_average_map(map_array=mask_array, figure_name="mean_mask", index=i, normalize=True)

            if args.celeb:
                save_path = os.path.join(
                    os.path.abspath(args.results_dir), 'average_masks/',
                        f"Celeb_{args.target_label}_{mask_group}.pdf",
                )
            if args.coco:
                save_path = os.path.join(
                    os.path.abspath(args.results_dir), 'average_masks/',
                        f"COCO_{args.target_label}_{mask_group}.pdf",
                )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            figure.savefig(save_path, dpi=150, bbox_inches="tight")

    if args.average_maps:
        print(f"Running average maps, saving to {os.path.join(os.path.abspath(args.results_dir), 'average_maps/')}")
        if scorer.multi_target:
            if args.subgroup_targets is not None:
                targets = args.subgroup_targets
            if args.one_sided_target is not None and args.one_sided_target != "Male":
                targets = ["Male", args.one_sided_target]
            elif args.average_maps_groups is not None:
                targets = [args.average_maps_groups[0]]
            else:
                targets = scorer.target_label
            for i, target in enumerate(targets):
                print(f"Calculating average map for {target}")
                attentions = []
                labels = []
                corrects = []
                if args.score_sample:
                    load_path = scorer.get_model_path(scorer.multi_target, args, multiple=True)
                    
                    for i, file in enumerate(glob.glob(load_path)):
                        print(f"Loading model from {file}")
                        model_num = re.search(r"([0-9]+?)(\.pt)$", file).group(1)
                        model.load_state_dict(torch.load(file))

                        attention_array, label_array, correct_array = scorer.get_all_attentions(target)
                        attentions.append(attention_array)
                        labels.append(label_array)
                        corrects.append(correct_array)
                        
                    attention_array = torch.cat(attentions, axis=0).cpu()
                    label_array = torch.cat(labels, axis=0).cpu()
                    correct_array = torch.cat(corrects, axis=0).cpu()
                else:
                    attention_array, label_array, correct_array = scorer.get_all_attentions(target)
                figure = scorer.calculate_average_map(map_array=attention_array, figure_name="mean_attention", index=i, normalize=True)
                
                if args.celeb:
                    save_path = os.path.join(
                    os.path.abspath(args.results_dir), 'average_maps/'
                        f"Celeb_{"-".join(args.target_label) if args.target_label != "all" else args.target_label}_{target}_{args.model}{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}{'_averaged' if args.score_sample else ''}.pdf",
                    )
                elif args.coco:
                    save_path = os.path.join(
                    os.path.abspath(args.results_dir), 'average_maps/'
                        f"COCO_{"-".join(args.target_label) if args.target_label != "all" else args.target_label}_{target}_{args.model}{'_averaged' if args.score_sample else ''}.pdf",
                    )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                figure.savefig(save_path, dpi=150, bbox_inches="tight")

                if args.average_maps_groups is not None:
                    true_true = torch.nanmean(attention_array[np.logical_and(scorer.get_target_labels(label_array, args.average_maps_groups[0]) == 1, scorer.get_target_labels(label_array, args.average_maps_groups[1]) == 1)].float(), axis=0, keepdim=True)
                    true_false = torch.nanmean(attention_array[np.logical_and(scorer.get_target_labels(label_array, args.average_maps_groups[0]) == 1, scorer.get_target_labels(label_array, args.average_maps_groups[1]) == 0)].float(), axis=0, keepdim=True)
                    false_true = torch.nanmean(attention_array[np.logical_and(scorer.get_target_labels(label_array, args.average_maps_groups[0]) == 0, scorer.get_target_labels(label_array, args.average_maps_groups[1]) == 1)].float(), axis=0, keepdim=True)
                    false_false = torch.nanmean(attention_array[np.logical_and(scorer.get_target_labels(label_array, args.average_maps_groups[0]) == 0, scorer.get_target_labels(label_array, args.average_maps_groups[1]) == 0)].float(), axis=0, keepdim=True)
                    
                    true_true_fig = scorer.calculate_average_map(map_array=true_true, figure_name="mean_attention", index=i, normalize=True)
                    true_false_fig = scorer.calculate_average_map(map_array=true_false, figure_name="mean_attention", index=i, normalize=True)
                    false_true_fig = scorer.calculate_average_map(map_array=false_true, figure_name="mean_attention", index=i, normalize=True)
                    false_false_fig = scorer.calculate_average_map(map_array=false_false, figure_name="mean_attention", index=i, normalize=True)
                    true_true_fig.savefig(os.path.join(
                    os.path.abspath(args.results_dir), 'average_maps/'
                        f"Celeb_{"-".join(args.target_label) if args.target_label != "all" else args.target_label}_{args.average_maps_groups}_true_true{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}{'_averaged' if args.score_sample else ''}.pdf",
                    ), dpi=150, bbox_inches="tight")
                    true_false_fig.savefig(os.path.join(
                    os.path.abspath(args.results_dir), 'average_maps/'
                        f"Celeb_{"-".join(args.target_label) if args.target_label != "all" else args.target_label}_{args.average_maps_groups}_true_false{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}{'_averaged' if args.score_sample else ''}.pdf",
                    ), dpi=150, bbox_inches="tight")
                    false_true_fig.savefig(os.path.join(
                    os.path.abspath(args.results_dir), 'average_maps/'
                        f"Celeb_{"-".join(args.target_label) if args.target_label != "all" else args.target_label}_{args.average_maps_groups}_false_true{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}{'_averaged' if args.score_sample else ''}.pdf",
                    ), dpi=150, bbox_inches="tight")
                    false_false_fig.savefig(os.path.join(
                    os.path.abspath(args.results_dir), 'average_maps/'
                        f"Celeb_{"-".join(args.target_label) if args.target_label != "all" else args.target_label}_{args.average_maps_groups}_false_false{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}{'_averaged' if args.score_sample else ''}.pdf",
                    ), dpi=150, bbox_inches="tight")
        else:
            attentions = []
            if args.score_sample:
                load_path = scorer.get_model_path(scorer.multi_target, args, multiple=True)
                
                for file in glob.glob(load_path):
                    print(f"Loading model from {file}")
                    model_num = re.search(r"([0-9]+?)(\.pt)$", file).group(1)
                    model.load_state_dict(torch.load(file))

                    attention_array, label_array, correct_array = scorer.get_all_attentions()
                    attentions.append(attention_array.cpu())
                attention_array = torch.cat(attentions, axis=0)

            figure = scorer.calculate_average_map(map_array=attention_array, figure_name="mean_attention", normalize=True)

            if args.celeb:
                save_path = os.path.join(
                os.path.abspath(args.results_dir), 'average_maps/',
                    f"Celeb_{args.model}_{args.target_label}{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}{'_averaged' if args.score_sample else ''}.pdf",
                )
            elif args.waterbirds:
                save_path = os.path.join(
                os.path.abspath(args.results_dir), 'average_maps/',
                    f"Waterbirds{args.waterbirds_percentage}_{args.model}{'_averaged' if args.score_sample else ''}.pdf",
                )
            elif args.coco:
                save_path = os.path.join(
                os.path.abspath(args.results_dir), 'average_maps/',
                    f"COCO_{args.model}_{args.target_label}{'_averaged' if args.score_sample else ''}.pdf",
                )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            figure.savefig(save_path, dpi=150, bbox_inches="tight")

    if args.evaluate_model:
        if args.score_sample:
            
            load_path = trainer.get_model_path(trainer.multi_target, args, multiple=True)
            
            for file in glob.glob(load_path):
                print(f"Loading model from {file}")
                model_num = re.search(r"([0-9]+?)(\.pt)$", file).group(1)
                model.load_state_dict(torch.load(file))

                results, _, _ = trainer.evaluate_model(trainer.test_loader)

                if args.coco:
                    probs_dict = trainer.evaluate_model_probabilities(trainer.test_loader)

 
                if args.waterbirds:
                    save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-waterbirds{args.waterbirds_percentage}_{args.model}_{model_num}.csv')
                if args.celeb:
                    if trainer.multi_target and args.target_label != "all":
                        save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-celeb_{"-".join(args.target_label)}{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}_{args.model}_{model_num}.csv')
                    else:
                        save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-celeb_{args.target_label}{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}_{args.model}_{model_num}.csv')
                if args.coco:
                    if trainer.multi_target and args.target_label != "all":
                        save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-coco_{"-".join(args.target_label)}{args.model}_{model_num}.csv')
                        save_path_dict = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'probs_dict-coco_{"-".join(args.target_label)}{args.model}_{model_num}.pkl')
                    else:
                        save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-coco_{args.target_label}_{args.model}_{model_num}.csv')
                        save_path_dict = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'probs_dict-coco_{args.model}_{model_num}.pkl')

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                print(f"Saving evaluations to {save_path}")
                np.savetxt(save_path, results, delimiter=',', fmt='%d')
                if args.coco:
                    print(f"Saving probability dictionary to {save_path_dict}")
                    with open(save_path_dict, 'wb+') as handle:
                        pickle.dump(probs_dict, handle)

        else:
            results, _, _ = trainer.evaluate_model(trainer.test_loader)

            if args.coco:
                probs_dict = trainer.evaluate_model_probabilities(trainer.test_loader)

            if args.waterbirds:
                save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-waterbirds{args.waterbirds_percentage}_{args.model}.csv')
            if args.celeb:
                if trainer.multi_target and args.target_label != "all":
                    save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-celeb_{"-".join(args.target_label)}{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}_{args.model}.csv')
                else:
                    save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-celeb_{args.target_label}{"_mcc" + str(args.subgroup_train_mcc) + "-".join(args.subgroup_targets) if args.subgroup_train_mcc is not None else ""}{"_one-sided-" + args.one_sided_train + "-" + args.one_sided_target if args.one_sided_train is not None else ""}_{args.model}.csv')
            if args.coco:
                if trainer.multi_target and args.target_label != "all":
                    save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-coco_{"-".join(args.target_label)}_{args.model}.csv')
                    save_path_dict = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'probs_dict-coco_{"-".join(args.target_label)}{args.model}.pkl')

                else:
                    save_path = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'evaluations-coco_{args.target_label}_{args.model}.csv')
                    save_path_dict = os.path.join(os.path.abspath(args.results_dir), 'evaluations/', f'probs_dict-coco_{args.model}.pkl')


            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print(f"Saving evaluations to {save_path}")
            np.savetxt(save_path, results, delimiter=',', fmt='%d')
            if args.coco:
                with open(save_path_dict, 'wb+') as handle:
                    pickle.dump(probs_dict, handle)

    if args.test_accuracy:
        if args.score_sample:
            load_path = trainer.get_model_path(trainer.multi_target, args, multiple=True)
            
            average_precisions = []
            accuracies = []
            overall_ap = []
            overall_accuracies = []
            group_accuracies = []
            for i, file in enumerate(glob.glob(load_path)):
                print(f"Loading model from {file}")
                model_num = re.search(r"([0-9]+?)(\.pt)$", file).group(1)
                model.load_state_dict(torch.load(file))

                preds, labels, outputs = trainer.evaluate_model(trainer.test_loader)

                if args.waterbirds:
                    groups = trainer.get_groups(labels)
                    labels = trainer.get_target_labels(labels)

                sample_weights = compute_sample_weight(class_weight=dataset.train_class_weights, y=labels)
                test_accuracy = accuracy_score(labels.flatten(), np.round(preds).flatten(), sample_weight=np.repeat(sample_weights, labels.shape[1]) if trainer.multi_target else sample_weights)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    test_average_precision = average_precision_score(labels, preds, sample_weight=sample_weights, average="macro")

                overall_ap.append(test_accuracy)
                overall_accuracies.append(test_average_precision)
                if args.celeb or args.coco:
                    average_precision, accuracy, group_accuracy, group_sizes = trainer.per_class_accuracy(outputs, labels, confounder="Male")
                    average_precisions.append(average_precision)
                    accuracies.append(accuracy)
                    group_accuracies.append(group_accuracy)
                if args.waterbirds:
                    group_accuracy = []
                    for j in range(4):
                        group_accuracy.append(accuracy_score(labels[groups==j].flatten(), np.round(preds[groups==j]).flatten()))
                    group_accuracies.append(group_accuracy)
                
            ap_mean = np.mean(average_precisions, axis=0)
            ap_std = np.std(average_precisions, axis=0)
            if args.celeb or args.coco:
                accuracy_mean = np.mean(accuracies, axis=0)
                accuracy_std = np.std(accuracies, axis=0)
                group_accuracy_mean = np.mean(group_accuracies, axis=0)
                group_accuracy_std = np.std(group_accuracies, axis=0)
            if args.waterbirds:
                # group_ap_mean = np.mean(group_aps, axis=0)
                # group_ap_std = np.std(group_aps, axis=0)
                group_accuracy_mean = np.mean(group_accuracies, axis=0)
                group_accuracy_std = np.std(group_accuracies, axis=0)
                for i in range(4):
                    print(f"Group {i}: Accuracy {group_accuracy_mean[i]:.4f} ± {group_accuracy_std[i]:.4f}")

            print(f"Test Set - AP: AP {np.mean(overall_ap):.4f} ± {np.std(overall_ap):.4f} | Accuracy {np.mean(overall_accuracies):.4f} ± {np.std(overall_accuracies):.4f}")
            if args.celeb or args.coco:
                for i, target in enumerate(trainer.target_label):
                    percent_mask = group_sizes[i] >= sum(group_sizes[i]) * .01
                    arg = np.argmin(group_accuracy_mean[i, percent_mask])
                    worst_group_accuracy_mean = group_accuracy_mean[i, np.arange(group_accuracy_mean.shape[1])[percent_mask][arg]]
                    worst_group_accuracy_std = group_accuracy_std[i, np.arange(group_accuracy_mean.shape[1])[percent_mask][arg]]
                    print(f"{target}: AP {ap_mean[i]:.4f} ± {ap_std[i]:.4f} | Accuracy {accuracy_mean[i]:.4f} ± {accuracy_std[i]:.4f} | Worst Group Accuracy {worst_group_accuracy_mean:.4f} ± {worst_group_accuracy_std:.4f}")
        else:
            preds = trainer.evaluate_model(trainer.test_loader)

    if args.directional_bias_amplification is not None:
        if args.score_sample:
            load_path = trainer.get_model_path(trainer.multi_target, args, multiple=True)
            
            values_ta = []
            values_at = []
            average_precisions = []
            accuracies = []
            for file in glob.glob(load_path):
                print(f"Loading model from {file}")
                model_num = re.search(r"([0-9]+?)(\.pt)$", file).group(1)
                model.load_state_dict(torch.load(file))

                preds, labels, outputs = trainer.evaluate_model(trainer.test_loader)

                average_precision, accuracy, _, _ = trainer.per_class_accuracy(outputs, labels)
                average_precisions.append(average_precision)
                accuracies.append(accuracy)

                attribute = dataset.LABEL_DICT[args.directional_bias_amplification]
                task_labels = labels
                task_preds = preds
                attribute_labels = np.expand_dims(labels[:, attribute], 1)
                attribute_preds = np.expand_dims(preds[:, attribute], 1)
                task_labels_train = dataset.label_array[dataset.train_idx]
                attribute_labels_train = np.expand_dims(dataset.label_array[dataset.train_idx][:, attribute], 1)
                names = [list(dataset.LABEL_DICT.keys()), [args.directional_bias_amplification]]
                values_ta.append(biasamp_task_to_attribute(task_labels, attribute_labels, attribute_preds, task_labels_train, attribute_labels_train))
                values_at.append(biasamp_attribute_to_task(task_labels, attribute_labels, task_preds, task_labels_train, attribute_labels_train))
                
            ap_mean = np.mean(average_precisions, axis=0)
            ap_std = np.std(average_precisions, axis=0)
            accuracy_mean = np.mean(accuracies, axis=0)
            accuracy_std = np.std(accuracies, axis=0)

            for i, target in enumerate(trainer.target_label):
                print(f"{target}: AP {ap_mean[i]:.4f} ± {ap_std[i]:.4f} | Accuracy {accuracy_mean[i]:.4f} ± {accuracy_std[i]:.4f}")

            ta_mean = np.mean(values_ta, axis=0)
            ta_std = np.std(values_ta, axis=0)

            print("Task to Attribute")
            sorted_indices = np.argsort(ta_mean.flatten())
            for i in sorted_indices[::-1]:
                a, t = i // len(names[0]), i % len(names[0])
                print("{0} - {1}: {2:.4f} ± {3:.4f}".format(names[1][a], names[0][t], ta_mean[a][t], ta_std[a][t]))

            at_mean = np.mean(values_at, axis=0)
            at_std = np.std(values_at, axis=0)

            print("Attribute to Task")
            sorted_indices = np.argsort(at_mean.flatten())
            for i in sorted_indices[::-1]:
                a, t = i // len(names[0]), i % len(names[0])
                print("{0} - {1}: {2:.4f} ± {3:.4f}".format(names[1][a], names[0][t], at_mean[a][t], at_std[a][t]))
        else:
            raise NotImplementedError("Directional bias amplifcation metric has not been implemented for only one model yet")
            


if __name__ == '__main__':
    main()
