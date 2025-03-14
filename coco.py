from operator import itemgetter
import os
from typing import Union
import glob
from PIL import Image
import time
import pickle

from tqdm import tqdm
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import DataLoader

from bias import BiasBase
from bias_dataset import BiasDataset
from bias_trainer import BiasTrainer
from bias_scorer import BiasScorer
from ap_utils import normalized_ap_wrapper

class COCOBase(BiasBase):

    BAD_CLASSES = ['street sign', 'hat', 'shoe', 'eye glasses', 'plate', 'mirror',
            'window', 'desk', 'door', 'blender', 'hair brush']

    def __init__(self, target_label: Union[str, list[str]], target_masks: list[str], label_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        labels_file = os.path.join(label_dir, "labels.txt")
        self.text_labels, self.LABEL_DICT = self.make_label_dict(labels_file)
        self.INDEX_DICT = dict((v,k) for k,v in self.LABEL_DICT.items())

        if target_label == "all":
            self.target_label = list(self.LABEL_DICT.keys())
        else:
            self.target_label = target_label
        self.target_masks = [self.LABEL_DICT[target_mask] for target_mask in target_masks]
        self.multi_target = not isinstance(self.target_label, str)

        self.MASK_GROUPS = {x:[x] for x in self.LABEL_DICT.keys()}

    def make_label_dict(self, labels_file):
        labels_txt = open(labels_file, 'r').read().split('\n')

        labels = {}
        for i in range(1, 183):
            labels[i] = labels_txt[i].split(' ', 1)[1]

        # Remove 'bad' classes based on https://github.com/nightrome/cocostuff/blob/master/labels.md
        labels_171 = [x for x in labels.values() if x not in self.BAD_CLASSES]

        labels_to_onehot = {}
        for i in range(171):
            labels_to_onehot[labels_171[i]] = i

        return labels, labels_to_onehot
    
    def get_masks(self, masks, mask_group=None):
        if mask_group:
            target_masks = [self.LABEL_DICT[target_mask] for target_mask in mask_group]
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
            if len(self.target_label) == 171:
                return ["Image {0}, Label all".format(i) for i in range(16)]
            else:
                return ["Image {0}, Label {1}".format(i, self.target_label) for i in range(16)]
                # return ["Image {0}, Label {1}: {2}".format(i, self.target_label, self.get_target_labels(labels)[i].bool().tolist()) for i in range(16)]
        else:
            return ["Image {0}, Label {1}: {2}".format(i, self.target_label, bool(self.get_target_labels(labels)[i])) for i in range(16)]
        
    def get_model_path(self, multi_target, args, index=None, multiple=False):
        folder_path = args.save_model if hasattr(args, 'save_mode') and args.save_model else args.load_model
        if multi_target and args.target_label != "all":
            load_path = os.path.join(os.path.abspath(folder_path), f'coco_{"-".join(args.target_label)}_{args.model}{"_" + str(index) if index is not None else ""}{"_*" if multiple else ""}.pt')
        else:
            load_path = os.path.join(os.path.abspath(folder_path), f'coco_{args.target_label}_{args.model}{"_" + str(index) if index is not None else ""}{"_*" if multiple else ""}.pt')
        
        return load_path


class COCODataset(COCOBase, BiasDataset):

    def __init__(
        self,
        data_dir: str,
        target_resolution: tuple[int, int],
        target_label: Union[str, list[str]],
        target_masks: list[str],
        augment_data: bool = False,
        seed: int = 0,
    ):
        super().__init__(
            label_dir=data_dir,
            data_dir=data_dir,
            target_resolution=target_resolution,
            augment_data=augment_data,
            target_label=target_label,
            target_masks=target_masks,
        )

        self.image_dir = os.path.join(data_dir, "coco2014")
        self.annotations_dir = os.path.join(data_dir, "annotations")

        anno_train = sorted(glob.glob(os.path.join(self.annotations_dir, 'train2017/*.png')))
        anno_val = sorted(glob.glob(os.path.join(self.annotations_dir, 'val2017/*.png')))
        self.anno_files = anno_train + anno_val

        train = sorted(glob.glob(os.path.join(self.image_dir, "train2014/*.jpg")))
        val = sorted(glob.glob(os.path.join(self.image_dir, "val2014/*.jpg")))

        print('anno_train {}, anno_val {}, train {}, val {}'.format(len(anno_train), len(anno_val), len(train), len(val)))

        train_labels_path = os.path.join(data_dir, 'labels_train.pkl')
        val_labels_path = os.path.join(data_dir, 'labels_val.pkl')

        if os.path.isfile(train_labels_path):
            print(f"Loading processed COCO training labels from {train_labels_path}")
            train_dict = pickle.load(open(train_labels_path, 'rb'))
        else:
            train_dict = self.process_labels(train, "train")
            print(f"Saving processed COCO training labels to {train_labels_path}")
            with open(os.path.join(data_dir, 'labels_train.pkl'), 'wb+') as handle:
                pickle.dump(train_dict, handle)
        if os.path.isfile(val_labels_path):
            print(f"Loading processed COCO validation labels from {val_labels_path}")
            val_dict = pickle.load(open(val_labels_path, 'rb'))
        else:
            val_dict = self.process_labels(val, "val")
            print(f"Saving processed COCO validation labels to {val_labels_path}")
            with open(os.path.join(data_dir, 'labels_val.pkl'), 'wb+') as handle:
                pickle.dump(val_dict, handle)
            
        np.random.seed(seed)
        torch.manual_seed(seed)
        num_train_images = len(train_dict)
        indices = np.random.permutation(num_train_images)
        self.train_idx = indices[: int(num_train_images * 0.8)]
        self.val_idx = indices[int(num_train_images * 0.8) :]
        self.test_idx = list(range(num_train_images, num_train_images+len(val_dict)))

        trainval_split_array = np.zeros(num_train_images)
        trainval_split_array[self.train_idx] = 0
        trainval_split_array[self.val_idx] = 1

        self.split_array = np.concatenate((trainval_split_array, np.full((len(val_dict)), 2, dtype=int)))

        self.filename_array = np.concatenate((list(train_dict.keys()), list(val_dict.keys())))
        self.label_array = torch.cat((torch.stack(list(train_dict.values())), torch.stack(list(val_dict.values()))))

        self.loss_weights = self.get_loss_weights(self.label_array[self.train_idx].numpy())
        self.train_class_weights = self.get_class_weights(self.label_array[self.train_idx].numpy())

    def process_labels(self, anno_set_files, set_name):
        # Process COCO-2014 train/validation set labels:
        # 1. Remove class 'unlabeled'
    # 2. Replace COCO-2014 labels that only have things annotations withp
        #    COCO-2017 things and stuff annotations
        # 3. One-hot encode to [0-170]
        start_time = time.time()

        count = 0
        labels = {}

        pbar = iter(anno_set_files)
        pbar = tqdm(pbar, desc=f'Processing COCO-Stuff labels on {set_name} set', total=len(anno_set_files))

        for file in pbar:
            # COCO-2014 validation images can be in COCO-2017 train or validation
            anno_train_file = file.replace(f"coco2014/{set_name}2014/COCO_{set_name}2014_", "annotations/train2017/")
            anno_train_file = anno_train_file.replace("jpg", "png")
            anno_val_file = anno_train_file.replace("train2017", "val2017")

            # Open the correct things+stuff annotation file
            if anno_train_file in self.anno_files:
                anno_image = Image.open(anno_train_file)
            else:
                anno_image = Image.open(anno_val_file)

            # Process the COCO-2017 things+stuff annotations
            label = list(np.unique(np.array(anno_image)).astype(np.int16))
            if 255 in label:
                label.remove(255)  # Remove class 'unlabeled'
            label = [self.text_labels[k + 1] for k in label]  # Convert to human-readable labels
            label = [s for s in label if s not in self.BAD_CLASSES]  # Remove bad labels
            label = [self.LABEL_DICT[s] for s in label]  # Map labels to [0-170]
            label_onehot = torch.nn.functional.one_hot(torch.LongTensor(label), num_classes=171)
            label_onehot = label_onehot.sum(dim=0)
            labels[file] = label_onehot  # Save the one-hot encoded label

            # count += 1
            # if count % 1000 == 0:
            #     print(count, time.time() - start_time)

        print(f'Finished processing {len(labels)} {set_name} labels')
        return labels

    def __len__(self):
        return len(self.label_array)

    def get_img_name(self, idx: int):
        return self.filename_array[idx]

    def get_mask_name(self, idx: int):
        anno_train_file = self.filename_array[idx].replace(
            "coco2014/train2014/COCO_train2014_", "annotations/train2017/")
        anno_train_file = anno_train_file.replace("coco2014/val2014/COCO_val2014_", "annotations/train2017/")
        anno_train_file = anno_train_file.replace("jpg", "png")
        anno_val_file = anno_train_file.replace("train2017", "val2017")
        if anno_train_file in self.anno_files:
            return anno_train_file
        else:
            return anno_val_file

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


class COCOTrainer(COCOBase, BiasTrainer):

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
        label_dir: str,
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
            figure_frequency=figure_frequency,
            label_dir=label_dir,
        )
    
    def per_class_accuracy(self, pred_array, label_array, print_results=False, confounder = "Male"):
        average_precision = normalized_ap_wrapper(label_array, pred_array)
        average_precision2 = average_precision_score(label_array, pred_array, average=None)
        accuracy = np.zeros(len(self.target_label))
        for i, target in enumerate(self.target_label):
            sample_weights = compute_sample_weight(class_weight=self.dataset.train_class_weights[i], y=label_array[:, i])
            # average_precision[i] = average_precision_score(label_array, pred_array, sample_weight=sample_weights, average=None)[i]
            accuracy[i] = accuracy_score(label_array[:, i], np.round(pred_array[:, i]), sample_weight=sample_weights)
            accuracy2 = accuracy_score(label_array[:, i], np.round(pred_array[:, i]))
            if print_results is True:
                print(f"{target} Test Result: Positive Labels {np.sum(label_array[:, i])} | Accuracy {accuracy[i]:.4f} | Accuracy Unweighted {accuracy2:.4f} | Average Precision: {average_precision[i]:.4f} | Average Precision Unweighted: {average_precision2[i]:.4f}")
        return average_precision, accuracy
    
    def evaluate_model_probabilities(self, loader):
        self.model.eval()

        pbar = enumerate(loader)
        pbar = tqdm(pbar, desc='Running probability evaluation', total=len(loader))

        probs_dict = {}
        with torch.no_grad():
            for i_batch, (images, masks, labels, idxs) in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                probabilities = torch.sigmoid(outputs).squeeze().data.cpu().numpy()

                for j in range(images.shape[0]):
                    idx = idxs[j].item()
                    probs_dict[idx] = probabilities[j]

        return probs_dict


class COCOScorer(COCOBase, BiasScorer):

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
        label_dir,
        num_samples=10000,
        multiple_figures=True,
        figure_frequency=0,
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
            resize_attentions=resize_attentions,
            label_dir=label_dir
        )
        self.confounder_label = confounder_label

    def get_confounder_labels(self, labels, confounder_label=None):
        if confounder_label is None:
            return labels[:, self.LABEL_DICT[self.confounder_label]]
        return labels[:, self.LABEL_DICT[confounder_label]]

    def score_table(self, scores, labels, target1=None, target2=None, target=None, preds=None):
        true_true_scores = []
        true_false_scores = []
        false_true_scores = []
        false_false_scores = []

        if target: # mask score
            # mask exists, pred true
            true_true_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels).T, preds.T), scores, torch.nan))
            true_false_scores.append(torch.where(torch.logical_and(self.get_target_labels(labels).T, torch.logical_not(preds.T)), scores, torch.nan))
            false_true_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels).T), preds.T), scores, torch.nan))
            false_false_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(labels).T), torch.logical_not(preds.T)), scores, torch.nan))
            tile_length = 1
        elif self.multi_target and target1 and target2: # heatmap score
            true_true_scores.append(torch.where(torch.logical_and(self.get_target_labels(preds, target1), self.get_target_labels(preds, target2)), scores, torch.nan))
            true_false_scores.append(torch.where(torch.logical_and(self.get_target_labels(preds, target1), torch.logical_not(self.get_target_labels(preds, target2))), scores, torch.nan))
            false_true_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(preds, target1)), self.get_target_labels(preds, target2)), scores, torch.nan))
            false_false_scores.append(torch.where(torch.logical_and(torch.logical_not(self.get_target_labels(preds, target1)), torch.logical_not(self.get_target_labels(preds, target2))), scores, torch.nan))
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
        # target_masks = [self.LABEL_DICT[target_mask] for target_mask in self.target_masks]
        if len(all_scores.shape) == 1:
            all_scores = np.transpose(all_scores) 
        if is_mask_group:
            np.savetxt(os.path.join(os.path.dirname(self.save_directory), f'Masks-{"all" if len(self.target_label) == 171 else self.target_label}_{target}_{self.confounder_label}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv'), all_scores, delimiter=',', fmt='%.8f')
        elif self.multi_target and target1 and target2:
            np.savetxt(os.path.join(os.path.dirname(self.save_directory), f'Heatmap-{"all" if len(self.target_label) == 171 else self.target_label}_{target1}_{target2}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv'), all_scores, delimiter=',', fmt='%.8f')
        else:
            np.savetxt(os.path.join(os.path.dirname(self.save_directory), f'{self.target_label}_{self.target_masks}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv'), all_scores, delimiter=',', fmt='%.8f')
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
            return f"Target: {'all' if len(self.target_label) == 171 else self.target_label}, Subset: {subset}"
        else:
            return f"Target: {'all' if len(self.target_label) == 171 else self.target_label} / {self.target_label[index]}, Subset: {subset}"