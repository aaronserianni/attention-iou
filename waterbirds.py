import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from bias import BiasBase
from bias_dataset import BiasDataset
from bias_trainer import BiasTrainer
from bias_scorer import BiasScorer


class WaterbirdsBase(BiasBase):

    MASK_GROUPS = {
        "background": True,
        "bird": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multi_target = False

    def get_masks(self, masks, background=False):
        if background:
            return 1 - masks
        else:
            return masks

    def get_target_labels(self, labels):
        return labels[:, 0]

    def get_groups(self, labels):
        return labels[:, 1]
    
    def get_figure_title(self, labels):
        return ["Image {0}, Group {1}".format(i, self.get_groups(labels)[i]) for i in range(16)]

    def get_model_path(self, multi_target, args, index=None, multiple=False):
        folder_path = args.save_model if hasattr(args, 'save_mode') and args.save_model else args.load_model
        load_path = os.path.join(os.path.abspath(folder_path), f'waterbirds{args.waterbirds_percentage}_{args.model}{"_" + str(index) if index is not None else ""}{"_*" if multiple else ""}.pt')
        return load_path


class WaterbirdsDataset(WaterbirdsBase, BiasDataset):

    def __init__(
        self,
        data_dir: str,
        target_resolution: tuple[int, int],
        augment_data: bool = False,
    ):
        super().__init__(
            data_dir=data_dir,
            target_resolution=target_resolution,
            augment_data=augment_data,
        )

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))

        self.label_array = self.metadata_df["y"].values
        self.confounder_array = self.metadata_df["place"].values

        self.n_confounders = 1
        self.n_groups = pow(2, 2)

        # 0: landbird land, 1: landbird water, 2: waterbird land, 3: waterbird water
        self.group_array = (
            self.label_array * (self.n_groups / 2) + self.confounder_array
        ).astype("int")

        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values

        train_mask = self.split_array == self.split_dict["train"]
        train_indices = np.where(train_mask)[0]
        self.loss_weights = self.get_loss_weights(self.label_array[train_indices])
        self.train_class_weights = self.get_class_weights(self.label_array[train_indices])

    def __len__(self):
        return len(self.metadata_df)

    def get_img_name(self, idx: int):
        return os.path.join(self.data_dir, self.filename_array[idx])

    def get_mask_name(self, idx: int):
        return os.path.join(
            self.data_dir,
            self.filename_array[idx].split("/")[0],
            "mask",
            self.filename_array[idx].split("/")[-1],
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        image, mask, label, idx = super().__getitem__(idx)
        group = self.group_array[idx]

        return image, mask, torch.as_tensor((label, group)), idx
    
    def get_loss_weights(self, label_array):
        weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=label_array)
        return weights
    
    def get_class_weights(self, label_array):
        class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=label_array)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        return class_weight_dict


class WaterbirdsTrainer(WaterbirdsBase, BiasTrainer):

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
            multiple_figures=multiple_figures,
            figure_frequency=figure_frequency,
        )

    def evaluate_model(self, loader):
        self.model.eval()

        pbar = enumerate(loader)
        pbar = tqdm(pbar, desc='Running evaluation', total=len(loader))

        pred_list = []
        label_list = []

        with torch.no_grad():
            for i_batch, (images, masks, labels, index) in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                if self.multi_target:
                    preds = torch.sigmoid(outputs)
                else:
                    _, preds = torch.max(outputs, 1)

                label_list.append(labels.detach().cpu())
                pred_list.append(preds.detach().cpu())

        label_array = np.concatenate(label_list, axis=0)
        pred_array = np.concatenate(pred_list)

        return np.round(pred_array).astype(int), label_array, pred_array


class WaterbirdsScorer(WaterbirdsBase, BiasScorer):

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
        waterbirds_percentage,
        num_samples=10000,
        multiple_figures=True,
        figure_frequency=0,
        resize_attentions=True,
    ):
        super().__init__(
            model,
            test_loader,
            device,
            attribution_method,
            writer,
            save_directory,
            normalize_scores,
            positive_scores,
            num_samples,
            multiple_figures,
            figure_frequency,
            resize_attentions=resize_attentions
        )
        self.waterbirds_percentage = waterbirds_percentage

    def score_table(self, scores, labels, target1=None, target2=None, target=None, preds=None):
        same_scores = torch.where(torch.logical_or(self.get_groups(labels)==0, self.get_groups(labels)==3), scores, torch.nan)
        different_scores = torch.where(torch.logical_or(self.get_groups(labels)==1, self.get_groups(labels)==2), scores, torch.nan)
        return torch.stack((same_scores, different_scores, scores))

    def get_average_map_labeled(self, maps, labels):
        same_map = torch.where(torch.logical_or(self.get_groups(labels)==0, self.get_groups(labels)==3).unsqueeze(1).unsqueeze(1), maps.to(self.device), torch.nan).nanmean(axis=0, keepdim=True)
        different_map = torch.where(torch.logical_or(self.get_groups(labels)==1, self.get_groups(labels)==2).unsqueeze(1).unsqueeze(1), maps.to(self.device), torch.nan).nanmean(axis=0, keepdim=True)

        return [("same", same_map), ("different", different_map)]

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
        is_mask_group=False,
    ):
        os.makedirs(os.path.dirname(self.save_directory), exist_ok=True)
        if is_mask_group:
            print("Saving scores to: " + str(os.path.join(os.path.dirname(self.save_directory), f'Masks-waterbirds{self.waterbirds_percentage}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv')))
            np.savetxt(os.path.join(os.path.dirname(self.save_directory), f'Masks-waterbirds{self.waterbirds_percentage}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv'), np.transpose(all_scores), delimiter=',', fmt='%.8f')
        else:
            print("Saving scores to: " + str(os.path.join(os.path.dirname(self.save_directory), f'waterbirds{self.waterbirds_percentage}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv')))
            np.savetxt(os.path.join(os.path.dirname(self.save_directory), f'waterbirds{self.waterbirds_percentage}_{self.attribution_method}_{score_name}{"_normalized" if self.normalize_scores else ""}{"_positive" if self.positive_scores else ""}{"_averaged" if is_averaged else ""}_scores.csv'), all_scores, delimiter=',', fmt='%.8f')

        # print("Same accuracy")
        # print(np.nanmean(np.where(np.logical_or(self.get_groups(label_array)==0, self.get_groups(label_array)==3), correct_array, np.nan)))
        # print("Different accuracy")
        # print(np.nanmean(np.where(np.logical_or(self.get_groups(label_array)==1, self.get_groups(label_array)==2), correct_array, np.nan)))

    def get_average_map_title(self, subset:str):
        return f"Bias Percentage: {self.waterbirds_percentage}, Subset: {subset}"
