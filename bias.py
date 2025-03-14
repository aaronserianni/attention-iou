import abc
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torchvision import models
import torchvision.transforms.v2 as transforms

from pytorch_grad_cam.utils.image import show_cam_on_image

class BiasBase:

    multi_target: bool

    @abc.abstractmethod
    def get_masks(self, masks):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasBase, should implement 'get_target_masks'")

    @abc.abstractmethod
    def get_target_labels(self, labels):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasBase, should implement 'get_target_labels'")

    @abc.abstractmethod
    def get_target_value(self, target_name):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasBase, should implement 'get_target_value'")

    def invert_image(self, image):
        invNorm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

        image = invNorm(image)
        if len(image.shape) == 3:
            image = image.permute(1, 2, 0)
        else:
            image = image.permute(0, 2, 3, 1)
        return image

    def make_plot(self, images, titles: list[str], colors: list[str], multiple: bool = False, main_title:str = None):
        if multiple:
            fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
            for i in range(16):
                ax[i // 4, i % 4].imshow(images[i])
                ax[i // 4, i % 4].set_title(titles[i], color=colors[i], fontsize=8, wrap=True)
                ax[i // 4, i % 4].axis("off")
            if main_title:
                fig.suptitle(main_title, wrap=True)
        else:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], frameon=False, xticks=[], yticks=[])
            ax.axis("off")
            if titles[0] == "":
                ax.set_title(f"{main_title}", color=colors[0], fontsize=8, wrap=True)
            else:
                ax.set_title(f"{main_title} - {titles[0]}", color=colors[0], fontsize=8, wrap=True)
            ax.imshow(images[0])

        return fig

    def show_attention_on_image(self, images, maps):
        return [show_cam_on_image(image, map) for image, map in zip(images, maps)]

    @abc.abstractmethod
    def get_figure_title(self, labels):
        raise NotImplementedError(
            f"{self.__class__.__name__}, which is a subclass of BiasBase, should implement 'get_figure_title'"
        )

    def make_figure(
        self,
        images,
        labels=None,
        preds=None,
        masks=None,
        attentions=None,
        multiple=False,
        main_title=None,
        invert_images=True,
        title: str = None,
    ):
        if invert_images:
            images = self.invert_image(images)
        images = images.detach().cpu().numpy()

        if title is not None:
            titles = [title] * 16
        elif labels is not None:
            titles = self.get_figure_title(labels)
        else:
            titles = [""] * 16

        if preds is None:
            colors = ["black"] * 16
        else:
            colors = [
                ("green" if torch.all(preds[i] == self.get_target_labels(labels)[i]) else "red")
                for i in range(16)
            ]

        if attentions is not None:
            attentions = attentions.cpu().numpy()
            fig = self.make_plot(self.show_attention_on_image(images, 1 - attentions), titles, colors, multiple, main_title)
            # fig = self.make_plot(attentions, titles, colors, multiple)
        elif masks is not None:
            fig = self.make_plot(self.show_attention_on_image(images, 1 - self.get_masks(masks)), titles, colors, multiple, main_title)
        else:
            fig = self.make_plot(images, titles, colors, multiple, main_title)

        return fig

    @abc.abstractmethod
    def get_model_path(self, multi_target, args, index=None, multiple=False):
        assert not (index is not None and multiple)
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasBase, should implement 'get_model_name'")


class BCEWithLogitsLossFlipped(nn.Module):
    def __init__(
        self,
        pos_weight: Optional[Tensor] = None,
        flipped: bool = False,
    ) -> None:
        super(BCEWithLogitsLossFlipped, self).__init__()
        self.pos_weight = (1 / pos_weight if flipped else pos_weight) if pos_weight is not None else None
        self.flipped = flipped
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.flipped:
            return self.loss_function(-input, 1 - target)
        else:
            return self.loss_function(input, target)
        

class ResNet50(nn.Module):    
    def __init__(self, n_classes=1, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = models.resnet50(weights=(models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None))
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        outputs = self.fc(self.dropout(self.relu(features)))

        return outputs

def get_model(model_type, n_classes, weights):
    if model_type == "resnet18":
        model = models.resnet18(weights=(models.ResNet18_Weights.IMAGENET1K_V1 if weights else None))
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_classes)
    elif model_type == "resnet50":
        model = models.resnet50(weights=(models.ResNet50_Weights.IMAGENET1K_V2 if weights else None))
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_classes)
    elif model_type == "resnet50_extended":
        model = ResNet50(n_classes=n_classes, pretrained=weights)
    elif model_type == "efficientnet":
        model = models.efficientnet_v2_s(weights=(models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if weights else None))
        num_features = model.classifier[1].in_features
        model.classifier = nn.Linear(num_features, n_classes)
    elif model_type == "vit_b32":
        model = models.vit_b_32(weights=(models.ViT_B_32_Weights.IMAGENET1K_V1 if weights else None))
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, n_classes)
    else:
        raise ValueError(f"'{model_type}' is an invalid model for the argument --model, expected 'resnet18', 'resnet50', 'resnet50_extended', 'efficientnet', or 'vit_b32'")

    return model


def biasamp_task_to_attribute(task_labels, attribute_labels, attribute_preds, task_labels_train=None, attribute_labels_train=None, names=None):
    '''
    for each of the following, an entry of 1 is a prediction, and 0 is not
    task_labels: n x |T|, these are labels on the test set, where n is the number of samples, and |T| is the number of tasks to be classified
    attribute_labels: n x |A|, these are labels on the test set, where n is the number of samples, and |A| is the number of attributes to be classified
    attribute_preds: n x |A|, these are predictions on the test set for attribute

    optional: below are used for setting the direction of the indicator variable. if not provided, test labels are used
    task_labels_train: m x |T|, these are labels on the train set, where m is the number of samples, and |T| is the number of tasks to be classified
    attribute_labels_train: m x |A|, these are labels on the train set, where m is the number of samples, and |A| is the number of attributes to be classified

    names: list of [task_names, attribute_names]. if included, will print out the top 10 attribute-task pairs with the most bias amplification
    '''
    assert len(task_labels.shape) == 2 and len(attribute_labels.shape) == 2, 'Please read the shape of the expected inputs, which should be "num samples" by "num classification items"'
    if task_labels_train is None or attribute_labels_train is None:
        task_labels_train, attribute_labels_train = task_labels, attribute_labels
    num_t, num_a = task_labels.shape[1], attribute_labels.shape[1]
    
    # only include images that have attribute(s) and task(s) associated with it for calculation of indicator variable
    keep_indices = np.array(list(set(np.where(np.sum(task_labels_train, axis=1)>0)[0]).union(set(np.where(np.sum(attribute_labels_train, axis=1)>0)[0]))))
    task_labels_train, attribute_labels_train = task_labels_train[keep_indices], attribute_labels_train[keep_indices]
    
    # y_at calculation
    p_at = np.zeros((num_a, num_t))
    p_a_p_t = np.zeros((num_a, num_t))
    num_train = len(task_labels_train)
    for a in range(num_a):
        for t in range(num_t):
            t_indices = np.where(task_labels_train[:, t]==1)[0]
            a_indices = np.where(attribute_labels_train[:, a]==1)[0]
            at_indices = set(t_indices)&set(a_indices)
            p_a_p_t[a][t] = (len(t_indices)/num_train)*(len(a_indices)/num_train)
            p_at[a][t] = len(at_indices)/num_train
    y_at = np.sign(p_at - p_a_p_t)

    # delta_at calculation
    a_cond_t = np.zeros((num_a, num_t))
    ahat_cond_t = np.zeros((num_a, num_t))
    for a in range(num_a):
        for t in range(num_t):
            a_cond_t[a][t] = np.mean(attribute_labels[:, a][np.where(task_labels[:, t]==1)[0]])
            ahat_cond_t[a][t] = np.mean(attribute_preds[:, a][np.where(task_labels[:, t]==1)[0]])
    delta_at = ahat_cond_t - a_cond_t

    values = y_at*delta_at
    val = np.nanmean(values)

    if names is not None:
        assert len(names) == 2, "Names should be a list of the task names and attribute names"
        task_names, attribute_names = names
        assert len(task_names)==num_t and len(attribute_names)==num_a, "The number of names should match both the number of tasks and number of attributes"

        sorted_indices = np.argsort(np.absolute(values).flatten())
        for i in sorted_indices[::-1]:
            a, t = i // num_t, i % num_t
            print("{0} - {1}: {2:.4f}".format(attribute_names[a], task_names[t], values[a][t]))
    return values
     
def biasamp_attribute_to_task(task_labels, attribute_labels, task_preds, task_labels_train=None, attribute_labels_train=None, names=None):
    '''
    for each of the following, an entry of 1 is a prediction, and 0 is not
    task_labels: n x |T|, these are labels on the test set, where n is the number of samples, and |T| is the number of tasks to be classified
    attribute_labels: n x |A|, these are labels on the test set, where n is the number of samples, and |A| is the number of attributes to be classified
    task_preds: n x |T|, these are predictions on the test set for task

    optional: below are used for setting the direction of the indicator variable. if not provided, test labels are used
    task_labels_train: m x |T|, these are labels on the train set, where m is the number of samples, and |T| is the number of tasks to be classified
    attribute_labels_train: m x |A|, these are labels on the train set, where m is the number of samples, and |A| is the number of attributes to be classified

    names: list of [task_names, attribute_names]. if included, will print out the top 10 attribute-task pairs with the most bias amplification
    '''

    assert len(task_labels.shape) == 2 and len(attribute_labels.shape) == 2, 'Please read the shape of the expected inputs, which should be "num samples" by "num classification items"'
    if task_labels_train is None or attribute_labels_train is None:
        task_labels_train, attribute_labels_train = task_labels, attribute_labels
    num_t, num_a = task_labels.shape[1], attribute_labels.shape[1]
    
    # only include images that have attribute(s) and task(s) associated with it for calculation of indicator variable
    keep_indices = np.array(list(set(np.where(np.sum(task_labels_train, axis=1)>0)[0]).union(set(np.where(np.sum(attribute_labels_train, axis=1)>0)[0]))))
    task_labels_train, attribute_labels_train = task_labels_train[keep_indices], attribute_labels_train[keep_indices]
    
    # y_at calculation
    p_at = np.zeros((num_a, num_t))
    p_a_p_t = np.zeros((num_a, num_t))
    num_train = len(task_labels_train)
    for a in range(num_a):
        for t in range(num_t):
            t_indices = np.where(task_labels_train[:, t]==1)[0]
            a_indices = np.where(attribute_labels_train[:, a]==1)[0]
            at_indices = set(t_indices)&set(a_indices)
            p_a_p_t[a][t] = (len(t_indices)/num_train)*(len(a_indices)/num_train)
            p_at[a][t] = len(at_indices)/num_train
    y_at = np.sign(p_at - p_a_p_t)

    # delta_at calculation
    t_cond_a = np.zeros((num_a, num_t))
    that_cond_a = np.zeros((num_a, num_t))
    for a in range(num_a):
        for t in range(num_t):
            t_cond_a[a][t] = np.mean(task_labels[:, t][np.where(attribute_labels[:, a]==1)[0]])
            that_cond_a[a][t] = np.mean(task_preds[:, t][np.where(attribute_labels[:, a]==1)[0]])
    delta_at = that_cond_a - t_cond_a

    values = y_at*delta_at
    val = np.nanmean(values)

    if names is not None:
        assert len(names) == 2, "Names should be a list of the task names and attribute names"
        task_names, attribute_names = names
        assert len(task_names)==num_t and len(attribute_names)==num_a, "The number of names should match both the number of tasks and number of attributes"

        sorted_indices = np.argsort(np.absolute(values).flatten())
        for i in sorted_indices[::-1]:
            a, t = i // num_t, i % num_t
            print("{0} - {1}: {2:.4f}".format(attribute_names[a], task_names[t], values[a][t]))
    return values