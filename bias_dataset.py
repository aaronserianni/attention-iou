import abc
from typing import Union

from PIL import Image
import numpy as np

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader, Subset

from bias import BiasBase

def transform(image, mask, target_resolution: tuple[int, int], augment=False):
    scale = 256.0 / 224.0

    image = transforms.functional.resize(
        image,
        size=(int(target_resolution[0] * scale), int(target_resolution[1] * scale)),
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    mask = transforms.functional.resize(
        mask,
        size=(int(target_resolution[0] * scale), int(target_resolution[1] * scale)),
        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
    )

    if augment:
        image = transforms.functional.resize(image, size=target_resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(mask, size=target_resolution, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

        params = transforms.RandomResizedCrop.get_params(image, scale=(0.7, 1.0), ratio=(0.75, 1.3333333333333333))
        image = transforms.functional.resized_crop(image, *params, size=target_resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.resized_crop(mask, *params, size=target_resolution, interpolation=transforms.InterpolationMode.NEAREST_EXACT)

        if(torch.empty(1).uniform_() > 0.5):
            transforms.functional.horizontal_flip(image)
            transforms.functional.horizontal_flip(mask)
    else:
        image = transforms.functional.center_crop(image, output_size=target_resolution)
        mask = transforms.functional.center_crop(mask, output_size=target_resolution)

    image = transforms.functional.to_dtype(image, torch.float32, scale=True)
    mask = transforms.functional.to_dtype(torch.round(mask), torch.float32, scale=True)
    mask = transforms.functional.to_grayscale(mask).squeeze()

    image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return image, mask


class BiasDataset(Dataset):

    filename_array: list[str]
    label_array: list[int]
    split_array: list[int]

    loss_weights: np.typing.ArrayLike
    train_class_weights: np.typing.ArrayLike

    def __init__(
        self,
        data_dir: str,
        target_resolution: Union[int, tuple[int, int]],
        augment_data: bool = False,
    ):
        self.data_dir = data_dir
        self.split_dict = {"train": 0, "val": 1, "test": 2}

        self.target_resolution = (
            (target_resolution, target_resolution)
            if isinstance(target_resolution, int)
            else target_resolution
        )
        self.augment_data = augment_data
        self.transform = transform

    @abc.abstractmethod
    def __len__(self):
        return

    @abc.abstractmethod
    def get_img_name(self, idx: int) -> str:
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasDataset, should implement 'get_img_name'")

    @abc.abstractmethod
    def get_mask_name(self, idx: int) -> str:
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasDataset, should implement 'get_mask_name'")

    def __getitem__(self, idx: int) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.get_img_name(idx)
        image = transforms.functional.to_image(Image.open(img_name).convert('RGB'))

        mask_name = self.get_mask_name(idx)
        mask = transforms.functional.to_image(Image.open(mask_name).convert('RGB'))

        if self.split_array[idx] == self.split_dict['train']:
            image, mask = self.transform(image, mask, self.target_resolution, augment=self.augment_data)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']]):
            image, mask = self.transform(image, mask, self.target_resolution, augment=False)

        label = self.label_array[idx]

        return image, mask, label, idx

    def get_splits(self, splits: str) -> list[Subset]:
        subsets = {}
        for split in splits:
            assert split in ("train", "val", "test"), split + " is not a valid split"
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            subsets[split] = Subset(self, indices)
        return subsets

    def get_loader(self, train: bool, **kwargs) -> DataLoader:
        if not train:
            shuffle = False
        else:
            shuffle = True

        loader = DataLoader(self, shuffle=shuffle, **kwargs)
        return loader

    def prepare_dataloaders(
        self,
        train: bool = True,
        return_full_dataset: bool = False,
    ) -> tuple[list[Dataset], int]:
        if return_full_dataset:
            return self
        if train:
            splits = ["train", "val", "test"]
        else:
            splits = ["test"]

        subsets = self.get_splits(splits)
        bias_subsets = [DatasetWrapper(subsets[split]) for split in splits]
        return bias_subsets
    
    @abc.abstractmethod
    def get_loss_weights(self, label_array):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasDataset, should implement 'get_loss_weights'")
    
    @abc.abstractmethod
    def get_class_weights(self, label_array):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasDataset, should implement 'get_class_weights'")


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def input_size(self):
        for image, _, _ in self:
            return image.size()

    def get_loader(
        self,
        shuffle: bool,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        loader = DataLoader(
            self,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return loader
