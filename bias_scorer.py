import abc
import warnings

import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms.v2 as transforms

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from torchray.attribution.deconvnet import deconvnet
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.extremal_perturbation import (
    extremal_perturbation,
    contrastive_reward,
)
from torchray.attribution.gradient import gradient
from torchray.attribution.guided_backprop import guided_backprop
from torchray.attribution.linear_approx import linear_approx
from torchray.attribution.rise import rise_class

from bias import BiasBase

class BinaryClassClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        output = torch.abs(model_output) * self.category
        return output[output.nonzero().squeeze()]

class BinaryClassPositiveClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        output = model_output * self.category
        return output[output.nonzero().squeeze()]

def reshape_transform(tensor, height=7, width=7):
    patch_size = int((tensor.size(1) - 1) ** 0.5)
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        patch_size, patch_size, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

class BiasScorer(BiasBase):

    cams = {
        "GradCAM": GradCAM,
        "HiResCAM": HiResCAM,
        "ScoreCAM": ScoreCAM,
        "GradCAMPlusPlus": GradCAMPlusPlus,
        "AblationCAM": AblationCAM,
        "XGradCAM": XGradCAM,
        "EigenCAM": EigenCAM,
        "EigenGradCAM": EigenGradCAM,
        # "FullGrad": FullGrad,
        "LayerCAM": LayerCAM,
    }

    torchray = {
        "DeConvNet": deconvnet,
        "ExcitationBackprop": excitation_backprop,
        "ExtremalPerturbation": extremal_perturbation,
        "Gradient": gradient,
        "GuidedBackprop": guided_backprop,
        "LinearApprox": linear_approx,
        "RISE": rise_class
    }

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
        num_samples=10000,
        multiple_figures=True,
        figure_frequency=0,
        resize_attentions=True
    ) -> None:
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.attribution_method = attribution_method
        self.writer = writer
        self.save_directory = save_directory
        self.normalize_scores = normalize_scores
        self.positive_scores = positive_scores
        self.num_samples = num_samples
        self.multiple_figures = multiple_figures
        self.figure_frequency = figure_frequency
        self.resize_attentions = resize_attentions

        self.score_functions = {
            "element_wise": self.element_wise_score,
            # "element_abs": self.element_wise_abs_score,
            # "pointing_game": self.pointing_game_score,
            # "iou": self.iou_score,
        }
        self.attribution_number = None

    # Attention-IOU score, when normalize_scores=True
    def element_wise_score(self, attentions, masks):
        if not self.resize_attentions:
            masks = transforms.functional.resize(
                masks.float(),
                size=attentions.shape[-2:],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            masks /= (1e-7 + masks.max(1)[0].max(1)[0].unsqueeze(1).unsqueeze(1))

        if self.normalize_scores:
            normalized_attentions = attentions / attentions.sum(axis=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            normalized_masks = masks / masks.sum(axis=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            squared_average = torch.pow((normalized_attentions + normalized_masks) / 2, 2).sum(axis=(1, 2))
            element_scores = (normalized_attentions * normalized_masks).sum(axis=(1, 2)) / squared_average
        else:
            # basic dot product
            element_scores = (attentions * masks).mean(axis=(1, 2))
        return element_scores

    def element_wise_abs_score(self, attentions, masks):
        if not self.resize_attentions:
            masks = transforms.functional.resize(
                masks.float(),
                size=attentions.shape[-2:],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            masks /= (1e-7 + masks.max(1)[0].max(1)[0].unsqueeze(1).unsqueeze(1))

        if self.normalize_scores:
            normalized_attentions = attentions / attentions.sum(axis=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            normalized_masks = masks / masks.sum(axis=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            element_scores = 1 - torch.abs(normalized_attentions - normalized_masks).sum(axis=(1, 2)) / 2
        else:
            element_scores = torch.abs(attentions - masks).mean(axis=(1, 2))
        return element_scores

    def pointing_game_score(self, attentions, masks):
        if not self.resize_attentions:
            masks = transforms.functional.resize(
                masks.float(),
                size=attentions.shape[-2:],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            masks /= (1e-7 + masks.max(1)[0].max(1)[0].unsqueeze(1).unsqueeze(1))

        if self.normalize_scores:
            normalized_masks = masks / masks.sum(axis=(1, 2)).unsqueeze(-1).unsqueeze(-1)
            point_game_scores = torch.gather(normalized_masks.flatten(start_dim=1), 1, attentions.flatten(start_dim=1).argmax(dim=1).unsqueeze(0)).squeeze()
        else:
            point_game_scores = torch.gather(masks.flatten(start_dim=1), 1, attentions.flatten(start_dim=1).argmax(dim=1).unsqueeze(0)).squeeze() > 0.5
        return point_game_scores

    def iou_score(self, attentions, masks):
        if not self.resize_attentions:
            masks = transforms.functional.resize(
                masks.float(),
                size=attentions.shape[-2:],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            masks /= (1e-7 + masks.max(1)[0].max(1)[0].unsqueeze(1).unsqueeze(1))

        iou_score_list = []
        for i in range(attentions.shape[0]):
            top_attention = torch.zeros_like(attentions[i].flatten(), dtype=torch.bool)
            top_attention[torch.topk(attentions[i].flatten(), torch.count_nonzero(masks[i] > 0.5))[1]] = True
            iou_score_list.append(
                torch.logical_and(top_attention, masks[i].flatten() > 0.5).sum()
                / torch.logical_or(top_attention, masks[i].flatten() > 0.5).sum()
            )
        iou_scores = torch.tensor(iou_score_list, device=self.device)
        return iou_scores

    def get_attention_map(self, images, targets, model_output):
        if self.model.__class__.__name__ == 'ResNet':
            target_layers = [self.model.layer4[-1]]
        elif self.model.__class__.__name__ == 'EfficientNet':
            target_layers = [self.model.features[-1]]
        elif self.model.__class__.__name__ == 'VisionTransformer':
            target_layers = [self.model.encoder.layers[-1].ln_1]
        else:
            target_layers = [self.model.resnet.layer4[-1]]

        if self.attribution_method in self.cams:
            if self.multi_target:
                if self.positive_scores:
                    cam_targets = [BinaryClassPositiveClassifierOutputTarget(category) for category in targets]
                else:
                    cam_targets = [BinaryClassClassifierOutputTarget(category) for category in targets]
            else:
                cam_targets = None
            if self.attribution_method == "AblationCAM":
                cam = self.cams[self.attribution_method](
                    model=self.model,
                    target_layers=target_layers,
                    reshape_transform=(
                        reshape_transform
                        if self.model.__class__.__name__ == "VisionTransformer"
                        else None
                    ),
                    ablation_layer=AblationLayerVit() if self.attribution_method == "AblationCAM" else None)
            else:
                cam = self.cams[self.attribution_method](
                    model=self.model,
                    target_layers=target_layers,
                    reshape_transform=(
                        reshape_transform
                        if self.model.__class__.__name__ == "VisionTransformer"
                        else None
                    ))
            attentions = torch.tensor(cam(input_tensor=images, targets=cam_targets), device=self.device)
            del cam

        if self.attribution_method in self.torchray:
            if self.attribution_method in ["ExcitationBackprop", "LinearApprox"]:
                targets = targets if self.positive_scores else targets * torch.sign(model_output)
                attentions = self.torchray[self.attribution_method](self.model, images, targets, saliency_layer=target_layers[0]).detach()
            elif self.attribution_method == "RISE":
                attentions = self.torchray[self.attribution_method](self.model, images, target=targets)
            elif self.attribution_method == "ExtremalPerturbation":
                attentions = torch.cat([self.torchray[self.attribution_method](self.model, image.unsqueeze(0), label.item(), reward_func=contrastive_reward, areas=[0.12])[0] 
                                        for image, label in zip(images, targets)])
            else:
                targets = targets if self.positive_scores else targets * torch.sign(model_output)
                attentions = self.torchray[self.attribution_method](self.model, images, targets)
        if self.resize_attentions:
            attentions = transforms.functional.resize(
                torch.squeeze(attentions),
                size=images.shape[-2:],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        else:
            attentions = torch.squeeze(attentions)
        attentions -= attentions.min(1)[0].min(1)[0].unsqueeze(1).unsqueeze(1)
        attentions /= 1e-7 + attentions.max(1)[0].max(1)[0].unsqueeze(1).unsqueeze(1)
        attentions = torch.clamp(attentions, min=0, max=1)

        return attentions

    @abc.abstractmethod
    def score_table(self, scores, labels, target1=None, target2=None, preds=None):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasScorer, should implement 'score_table'")

    def sample_scores(self, score_array, correct_array, sample_indices, include_preds=False, correct=True):
        if include_preds:
            score_array = np.where(np.logical_not(np.logical_xor(correct, correct_array)), score_array, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            score_array_shape = list(score_array.shape)
            score_array_shape.insert(-1,-1)
            score_samples = np.nanmean(score_array[...,sample_indices].reshape(tuple(score_array_shape)), axis=-1)
            score_means = np.nanmean(score_samples, axis=-1)
            score_stds = np.nanstd(score_samples, axis=-1)
        return np.concatenate((score_means, score_stds), axis=0)

    def calculate_confounder_score(self, sample_scores=True):
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = True

        pbar = enumerate(self.test_loader)
        pbar = tqdm(pbar, desc=f'Running {self.attribution_method} score', total=len(self.test_loader))

        score_lists = {k: [] for k in self.score_functions.keys()}
        label_list = []
        pred_list = []
        correct_pred_list = []

        for i_batch, (images, masks, labels, idx) in pbar:
            images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            masks = self.get_masks(masks)

            attentions = self.get_attention_map(images, self.get_target_labels(labels), outputs)

            label_list.append(labels.cpu())

            if self.multi_target:
                preds = torch.round(torch.sigmoid(outputs)).int()
                correct_pred_list.append((torch.all(preds == self.get_target_labels(labels), dim=1)).cpu())
            else:
                _, preds = torch.max(outputs, 1)
                correct_pred_list.append((preds == self.get_target_labels(labels)).cpu())
            pred_list.append(preds.cpu())

            if (self.figure_frequency == 0 and i_batch == 0) or (
                self.figure_frequency != 0 and i_batch % self.figure_frequency == 0
            ):
                attentions = transforms.functional.resize(
                        torch.squeeze(attentions),
                        size=images.shape[-2:],
                        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
                    )

                self.writer.add_figure(
                    "attentions",
                    self.make_figure(images, labels, preds, attentions=attentions, multiple=self.multiple_figures, main_title=self.attribution_method),
                    global_step=(self.attribution_number if self.attribution_number else i_batch),
                )

            for key in self.score_functions.keys():
                scores = self.score_functions[key](attentions, masks)
                score_lists[key].append(self.score_table(scores, labels))

            del images
            del masks
            del labels
            del attentions

        score_arrays = {k: torch.concatenate(score_lists[k], axis=-1).cpu().numpy() for k in self.score_functions.keys()}
        correct_array = np.expand_dims(np.concatenate(correct_pred_list, axis=-1), axis=(0, 1))
        label_array = np.concatenate(label_list, axis=0)
        pred_array = np.concatenate(pred_list, axis=0)

        if sample_scores:
            indices = np.random.randint(
                score_arrays[list(score_arrays.keys())[0]].shape[-1],
                size=score_arrays[list(score_arrays.keys())[0]].shape[-1]
                * self.num_samples,
            )

            if len(score_arrays[list(score_arrays.keys())[0]].shape) == 2:
                all_scores = {k: np.vstack((self.sample_scores(score_arrays[k], correct_array, indices, include_preds=False, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=False)))
                                for k in self.score_functions.keys()}
            else:
                all_scores = {k: np.concatenate((self.sample_scores(score_arrays[k], correct_array, indices, include_preds=False, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=False)), 
                                                axis=0)
                                for k in self.score_functions.keys()}

            for key in self.score_functions.keys():
                self.save_scores(key, all_scores[key], label_array, correct_array)

            return all_scores
        else:
            return score_arrays, correct_array, label_array, pred_array

    def calculate_heatmap_score(self, target1, target2, sample_scores=True):
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = True

        pbar = enumerate(self.test_loader)
        pbar = tqdm(pbar, desc=f'Running {self.attribution_method} score', total=len(self.test_loader))

        score_lists = {k: [] for k in self.score_functions.keys()}
        label_list = []
        correct_pred_list = []
        pred_list = []

        for i_batch, (images, masks, labels, idx) in pbar:
            images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            masks = self.get_masks(masks)

            targets1 = torch.zeros_like(self.get_target_labels(labels))
            targets1[:, self.get_target_value(target1)] = 1
            targets2 = torch.zeros_like(self.get_target_labels(labels))
            targets2[:, self.get_target_value(target2)] = 1

            attentions1 = self.get_attention_map(images, targets1, outputs)
            attentions2 = self.get_attention_map(images, targets2, outputs)

            label_list.append(labels.cpu())

            if self.multi_target:
                preds = torch.round(torch.sigmoid(outputs)).int()
                correct_pred_list.append((torch.all(preds == self.get_target_labels(labels), dim=1)).cpu())
            else:
                _, preds = torch.max(outputs, 1)
                correct_pred_list.append((preds == self.get_target_labels(labels)).cpu())
            pred_list.append(preds.cpu())

            if (self.figure_frequency == 0 and i_batch == 0) or (
                self.figure_frequency != 0 and i_batch % self.figure_frequency == 0
            ):
                if not self.resize_attentions:
                    attentions1 = transforms.functional.resize(
                        torch.squeeze(attentions1),
                        size=images.shape[-2:],
                        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
                    )
                    attentions2 = transforms.functional.resize(
                        torch.squeeze(attentions2),
                        size=images.shape[-2:],
                        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
                    )

                self.writer.add_figure(
                    "attentions1",
                    self.make_figure(images, labels, preds, attentions=attentions1, multiple=self.multiple_figures, main_title=f"{self.attribution_method} - {target1}"),
                    global_step=(self.attribution_number if self.attribution_number else i_batch),
                )
                self.writer.add_figure(
                    "attentions2",
                    self.make_figure(images, labels, preds, attentions=attentions2, multiple=self.multiple_figures, main_title=f"{self.attribution_method} - {target2}"),
                    global_step=(self.attribution_number if self.attribution_number else i_batch),
                )

            for key in self.score_functions.keys():
                scores = self.score_functions[key](attentions1, attentions2)
                score_lists[key].append(self.score_table(scores, labels, target1=target1, target2=target2, preds=preds))

            del images
            del masks
            del labels
            del attentions1
            del attentions2

        score_arrays = {k: torch.concatenate(score_lists[k], axis=-1).cpu().numpy() for k in self.score_functions.keys()}
        correct_array = np.expand_dims(np.concatenate(correct_pred_list, axis=-1), axis=(0, 1))
        label_array = np.concatenate(label_list, axis=0)
        pred_array = np.concatenate(pred_list, axis=0)

        if sample_scores:
            indices = np.random.randint(
                score_arrays[list(score_arrays.keys())[0]].shape[-1],
                size=score_arrays[list(score_arrays.keys())[0]].shape[-1]
                * self.num_samples,
            )

            if len(score_arrays[list(score_arrays.keys())[0]].shape) == 2:
                all_scores = {k: np.vstack((self.sample_scores(score_arrays[k], correct_array, indices, include_preds=False, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=False)))
                                for k in self.score_functions.keys()}
            else:
                all_scores = {k: np.concatenate((self.sample_scores(score_arrays[k], correct_array, indices, include_preds=False, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=False)), 
                                                axis=0)
                                for k in self.score_functions.keys()}

            for key in self.score_functions.keys():
                self.save_scores(key, all_scores[key], label_array, correct_array, target1, target2)

            return all_scores
        else:
            return score_arrays, correct_array, label_array, pred_array

    def calculate_mask_score(self, mask_groups, target=None, sample_scores=True, uniform_attention=False):
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = True

        pbar = enumerate(self.test_loader)
        pbar = tqdm(pbar, desc=f'Running {self.attribution_method} score', total=len(self.test_loader))

        score_lists = {k: [] for k in self.score_functions.keys()}
        label_list = []
        correct_pred_list = []
        pred_list = []

        for i_batch, (images, masks, labels, idx) in pbar:
            images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)

            outputs = self.model(images)

            if target is not None:
                targets = torch.zeros_like(self.get_target_labels(labels))
                targets[:, self.get_target_value(target)] = 1
            else:
                targets = self.get_target_labels(labels)

            attentions = self.get_attention_map(images, targets, outputs)

            label_list.append(labels.cpu())

            if self.multi_target:
                preds = torch.round(torch.sigmoid(outputs)).int()
                correct_pred_list.append((torch.all(preds == self.get_target_labels(labels), dim=1)).cpu())
            else:
                _, preds = torch.max(outputs, 1)
                correct_pred_list.append((preds == self.get_target_labels(labels)).cpu())
            pred_list.append(preds.cpu())

            if (self.figure_frequency == 0 and i_batch == 0) or (
                self.figure_frequency != 0 and i_batch % self.figure_frequency == 0
            ):  
                if not self.resize_attentions:
                    attentions = transforms.functional.resize(
                        torch.squeeze(attentions),
                        size=images.shape[-2:],
                        interpolation=transforms.InterpolationMode.NEAREST_EXACT,
                    )

                if uniform_attention is True:
                    attentions = torch.ones_like(attentions)

                self.writer.add_figure(
                    "attentions",
                    self.make_figure(images, labels, preds, attentions=attentions, multiple=self.multiple_figures, main_title=f"{self.attribution_method} - {target}"),
                    global_step=(self.attribution_number if self.attribution_number else i_batch),
                )

            for key in self.score_functions.keys():
                mask_score_list = []
                for mask_group in mask_groups.values():
                    mask_score_list.append(self.score_functions[key](attentions, self.get_masks(masks, mask_group)).nan_to_num())
                score_lists[key].append(self.score_table(torch.stack(mask_score_list), labels, target=target, preds=preds))

            del images
            del masks
            del labels

        score_arrays = {k: torch.concatenate(score_lists[k], axis=-1).cpu().numpy() for k in self.score_functions.keys()}
        correct_array = np.expand_dims(np.concatenate(correct_pred_list, axis=-1), axis=(0, 1))
        label_array = np.concatenate(label_list, axis=0)
        pred_array = np.concatenate(pred_list, axis=0)

        if sample_scores:
            indices = np.random.randint(
                score_arrays[list(score_arrays.keys())[0]].shape[-1],
                size=score_arrays[list(score_arrays.keys())[0]].shape[-1]
                * self.num_samples,
            )

            if len(score_arrays[list(score_arrays.keys())[0]].shape) == 2:
                all_scores = {k: np.vstack((self.sample_scores(score_arrays[k], correct_array, indices, include_preds=False, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=False)))
                                for k in self.score_functions.keys()}
            else:
                all_scores = {k: np.concatenate((self.sample_scores(score_arrays[k], correct_array, indices, include_preds=False, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=True),
                                                self.sample_scores(score_arrays[k], correct_array, indices, include_preds=True, correct=False)), 
                                                axis=0)
                                for k in self.score_functions.keys()}

            for key in self.score_functions.keys():
                self.save_scores(key, all_scores[key], label_array, correct_array, is_mask_group=True, target=target)

            return all_scores
        else:
            return score_arrays, correct_array, label_array, pred_array

    @abc.abstractmethod
    def get_average_map_labeled(self, maps, labels):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasScorer, should implement 'get_average_map_labeled'")

    def get_all_attentions(self, attention_target = None):
        if self.multi_target and attention_target is None:
            raise ValueError("If model has multiple targets, 'attention_target' must be specified")

        for param in self.model.parameters():
            param.requires_grad = True

        pbar = enumerate(self.test_loader)
        pbar = tqdm(pbar, desc=f'Running {self.attribution_method}', total=len(self.test_loader))

        label_list = []
        correct_pred_list = []
        attention_list = []

        for i_batch, (images, masks, labels, idx) in pbar:
            images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)
            label_list.append(labels)

            outputs = self.model(images)
            masks = self.get_masks(masks)

            if self.multi_target:
                targets = torch.zeros_like(self.get_target_labels(labels))
                targets[:, self.get_target_value(attention_target)] = 1
            else:
                targets = self.get_target_labels(labels)

            attentions = self.get_attention_map(images, targets, outputs)
            attention_list.append(attentions)

            if self.multi_target:
                preds = torch.round(torch.sigmoid(outputs)).int()
                correct_pred_list.append((torch.all(preds == self.get_target_labels(labels), dim=1)).cpu())
            else:
                _, preds = torch.max(outputs, 1)
                correct_pred_list.append((preds == self.get_target_labels(labels)).cpu())

            del images
            del masks
            del labels
            del attentions

        label_array = torch.concatenate(label_list, axis=0)
        correct_array = torch.cat(correct_pred_list, axis=-1).unsqueeze(1).unsqueeze(1)
        attention_array = torch.cat(attention_list, axis=0)

        return attention_array, label_array, correct_array

    def get_all_masks(self, mask_group=None):
        label_list = []
        mask_list = []

        pbar = enumerate(self.test_loader)
        pbar = tqdm(pbar, desc=f"Running masks {mask_group}", total=len(self.test_loader))

        for i_batch, (images, masks, labels, idx) in pbar:
            images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)
            masks = self.get_masks(masks, mask_group=mask_group)

            label_list.append(labels)
            mask_list.append(masks)

            del images
            del masks
            del labels

        label_array = torch.concatenate(label_list, axis=0)
        mask_array = torch.cat(mask_list, axis=0)

        return mask_array, label_array

    def sample_attentions(self, attention_target=None, index=None):
        if self.multi_target and attention_target is None:
            raise ValueError("If model has multiple targets, 'attention_target' must be specified")

        for param in self.model.parameters():
            param.requires_grad = True

        images, masks, labels = next(iter(self.test_loader))
        images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)

        outputs = self.model(images)
        masks = self.get_masks(masks)

        if self.multi_target:
            targets = torch.zeros_like(self.get_target_labels(labels))
            targets[:, self.get_target_value(attention_target)] = 1
        else:
            targets = self.get_target_labels(labels)

        attentions = self.get_attention_map(images, targets, outputs)

        self.writer.add_figure(
                        "attentions",
                        self.make_figure(images, self.get_target_labels(labels, target_label=attention_target), preds=None, attentions=attentions, multiple=True, main_title=f"{self.attribution_method} - {attention_target}"),
                        global_step=0 if index is None else index,
                    )

    @abc.abstractmethod
    def get_average_map_title(self, subset):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasScorer, should implement 'get_average_map_title'")

    def calculate_average_map(self, map_array, figure_name, label_array=None, correct_array=None, index=None, normalize=False):
        if normalize:
            map_array = map_array / (1e-7 + map_array.sum(axis=(1, 2)).unsqueeze(-1).unsqueeze(-1))
        mean_map = torch.nanmean(map_array.float(), axis=0, keepdim=True)

        average_map_figure = self.make_figure(mean_map, multiple=False, main_title=self.attribution_method, invert_images=False, title=self.get_average_map_title("all"))
        self.writer.add_figure(
                        figure_name,
                        average_map_figure,
                        global_step=0 if index is None else index,
                    )

        if correct_array is not None:
            mean_map_correct = torch.nanmean(torch.where(torch.logical_not(torch.logical_xor(torch.ones_like(correct_array, dtype=torch.bool), correct_array)) ,map_array, torch.nan), axis=0, keepdim=True)
            mean_map_incorrect = torch.nanmean(torch.where(torch.logical_not(torch.logical_xor(torch.zeros_like(correct_array, dtype=torch.bool), correct_array)), map_array, torch.nan), axis=0, keepdim=True)

            self.writer.add_figure(
                figure_name,
                self.make_figure(
                    mean_map_correct,
                    multiple=False,
                    main_title=self.attribution_method,
                    invert_images=False,
                    title=self.get_average_map_title("correct predictions"),
                ),
                global_step=1,
            )

            self.writer.add_figure(
                figure_name,
                self.make_figure(
                    mean_map_incorrect,
                    multiple=False,
                    main_title=self.attribution_method,
                    invert_images=False,
                    title=self.get_average_map_title("incorrect predictions"),
                ),
                global_step=2,
            )

        if label_array is not None:
            for i, labeled_map in enumerate(
                self.get_average_map_labeled(map_array, label_array)
            ):
                self.writer.add_figure(
                    figure_name if index is None else figure_name + "_" + i,
                    self.make_figure(
                        labeled_map[1],
                        multiple=False,
                        main_title=self.attribution_method,
                        invert_images=False,
                        title=self.get_average_map_title(labeled_map[0]),
                    ),
                    global_step=(3 + i) if index is None else index,
                )
        return average_map_figure

    @abc.abstractmethod
    def save_scores(self, score_name, all_scores, label_array=None, correct_array=None, target1=None, target2=None, is_averaged=False):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasScorer, should implement 'save_scores'")
