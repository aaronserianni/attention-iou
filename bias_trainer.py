import abc
import warnings

from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

import torch
from torch.utils.data import DataLoader

from bias import BiasBase
from bias_dataset import BiasDataset


class BiasTrainer(BiasBase):

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
        multiple_figures = True,
        figure_frequency = 0
    ) -> None:
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.writer = writer
        self.multiple_figures = multiple_figures
        self.figure_frequency = figure_frequency

    def train_model(self, num_epochs: int = 10, validation_frequency: int = 1):
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            pbar = enumerate(self.train_loader)
            pbar = tqdm(pbar, desc=f'Running train epoch {epoch}', total=len(self.train_loader))

            ep_loss = 0

            label_list = []
            pred_list = []

            for i_batch, (images, masks, labels, idx) in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(images)

                if self.multi_target:
                    loss = self.loss_function(outputs, self.get_target_labels(labels).float())
                    preds = torch.sigmoid(outputs)
                else:
                    loss = self.loss_function(outputs, self.get_target_labels(labels))
                    _, preds = torch.max(outputs, 1)

                label_list.append(self.get_target_labels(labels).detach().cpu())
                pred_list.append(preds.detach().cpu())

                label_array = np.concatenate(label_list, axis=0)
                pred_array = np.concatenate(pred_list)

                loss.backward()
                ep_loss += loss
                self.optimizer.step()

                # sample_weights = compute_sample_weight(class_weight=self.dataset.train_class_weights, y=label_array) # TODO: Something about repeating input label array for accuracy? idk
                # accuracy = accuracy_score(label_array.flatten(), np.round(pred_array).flatten(), sample_weight=np.repeat(sample_weights, label_array.shape[1]) if self.multi_target else sample_weights)
                # with warnings.catch_warnings():
                #     warnings.simplefilter("ignore", category=UserWarning)
                #     average_precision = average_precision_score(label_array, pred_array, sample_weight=sample_weights, average="macro")

                pbar.set_postfix(
                    loss=f"{ep_loss.item() / (i_batch + 1):.4f}",
                    # accuracy=f"{accuracy:.4f}",
                    # AP=f"{average_precision:.4f}",
                )

                if (self.figure_frequency == 0 and i_batch == 0) or (
                    self.figure_frequency != 0 and i_batch % self.figure_frequency == 0
                ):
                    self.writer.add_figure('augmentation',
                                    self.make_figure(images, labels, masks=masks, preds=preds, multiple=self.multiple_figures),
                                    global_step=i_batch)

            ep_loss = ep_loss / len(self.train_loader)
            self.scheduler.step(ep_loss)

            sample_weights = compute_sample_weight(class_weight=self.dataset.train_class_weights, y=label_array) # TODO: Something about repeating input label array for accuracy? idk
            accuracy = accuracy_score(label_array.flatten(), np.round(pred_array).flatten(), sample_weight=np.repeat(sample_weights, label_array.shape[1]) if self.multi_target else sample_weights)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                average_precision = average_precision_score(label_array, pred_array, sample_weight=sample_weights, average="macro")

            print(f"Epoch {epoch}/{num_epochs} | Epoch Loss: {ep_loss:.4f} | Epoch Accuracy: {accuracy:.4f} | Epoch Average Precision: {average_precision:.4f}")

            self.writer.add_scalar('training loss',
                                    ep_loss,
                                    epoch)
            self.writer.add_scalar('training accuracy',
                                    accuracy,
                                    epoch)
            self.writer.add_scalar('training average precision',
                                    average_precision,
                                    epoch)

            if validation_frequency is not None:
                if epoch % validation_frequency == 0:
                    self.val_model(epoch, num_epochs)

        del accuracy
        del average_precision

    def val_model(self, epoch: int, num_epochs: int):
        self.model.eval()

        pbar = enumerate(self.val_loader)
        pbar = tqdm(pbar, desc=f'Running val epoch {epoch}', total=len(self.val_loader))

        total_val_loss = 0

        label_list = []
        pred_list = []

        with torch.no_grad():
            for i_batch, (images, masks, labels, idx) in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                if self.multi_target:
                    val_loss = self.loss_function(outputs, self.get_target_labels(labels).float())
                    preds = torch.sigmoid(outputs)
                else:
                    val_loss = self.loss_function(outputs, self.get_target_labels(labels))
                    _, preds = torch.max(outputs, 1)

                label_list.append(self.get_target_labels(labels).detach().cpu())
                pred_list.append(preds.detach().cpu())

                label_array = np.concatenate(label_list, axis=0)
                pred_array = np.concatenate(pred_list)

                total_val_loss += val_loss

        sample_weights = compute_sample_weight(class_weight=self.dataset.train_class_weights, y=label_array)
        val_accuracy = accuracy_score(label_array.flatten(), np.round(pred_array).flatten(), sample_weight=np.repeat(sample_weights, label_array.shape[1]) if self.multi_target else sample_weights)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            val_average_precision = average_precision_score(label_array, pred_array, sample_weight=sample_weights, average="macro")

        print(f"Epoch {epoch}/{num_epochs} | Val Loss: {total_val_loss/len(self.val_loader):.4f} | Val Accuracy: {val_accuracy:.4f} | Val Average Precision: {val_average_precision:.4f}")

        if epoch == num_epochs and self.multi_target:
            self.per_class_accuracy(pred_array, label_array, print_results=True)

        self.writer.add_scalar('validation loss', total_val_loss/len(self.val_loader), epoch)
        self.writer.add_scalar('validation accuracy', val_accuracy, epoch)
        self.writer.add_scalar('validation average precision', val_average_precision, epoch)

        del val_accuracy
        del val_average_precision

    def test_model(self):
        self.model.eval()

        pbar = enumerate(self.test_loader)
        pbar = tqdm(pbar, desc='Running test', total=len(self.test_loader))

        total_test_loss = 0

        pred_list = []
        label_list = []

        with torch.no_grad():
            for i_batch, (images, masks, labels, idx) in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                if self.multi_target:
                    test_loss = self.loss_function(outputs, self.get_target_labels(labels).float())
                    preds = torch.sigmoid(outputs)
                else:
                    test_loss = self.loss_function(outputs, self.get_target_labels(labels))
                    _, preds = torch.max(outputs, 1)

                label_list.append(self.get_target_labels(labels).detach().cpu())
                pred_list.append(preds.detach().cpu())

                label_array = np.concatenate(label_list, axis=0)
                pred_array = np.concatenate(pred_list)

                total_test_loss += test_loss

                if (self.figure_frequency == 0 and i_batch == 0) or (
                    self.figure_frequency != 0 and i_batch % self.figure_frequency == 0
                ):
                    self.writer.add_figure('masks',
                                        self.make_figure(images, labels, masks=masks, preds=preds, multiple=self.multiple_figures),
                                        global_step=i_batch)
                    self.writer.add_figure('images',
                                        self.make_figure(images, labels, preds=preds, multiple=self.multiple_figures), global_step=i_batch)
        
        sample_weights = compute_sample_weight(class_weight=self.dataset.train_class_weights, y=label_array)
        test_accuracy = accuracy_score(label_array.flatten(), np.round(pred_array).flatten(), sample_weight=np.repeat(sample_weights, label_array.shape[1]) if self.multi_target else sample_weights)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            test_average_precision = average_precision_score(label_array, pred_array, sample_weight=sample_weights, average="macro")

        print(f"Test Loss: {total_test_loss/len(self.test_loader):.4f} | Test Accuracy: {test_accuracy:.4f} | Test Average Precision: {test_average_precision:.4f}")

        if self.multi_target:
            self.per_class_accuracy(pred_array, label_array, print_results=True)

        self.writer.add_scalar("test loss", total_test_loss / len(self.test_loader), 1)
        self.writer.add_scalar('test accuracy', test_accuracy, 1)
        self.writer.add_scalar('test average precision', test_average_precision, 1)

        del test_accuracy
        del test_average_precision

    def evaluate_model(self, loader):
        self.model.eval()

        pbar = enumerate(loader)
        pbar = tqdm(pbar, desc='Running evaluation', total=len(loader))

        pred_list = []
        label_list = []

        with torch.no_grad():
            for i_batch, (images, masks, labels, idx) in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                if self.multi_target:
                    preds = torch.sigmoid(outputs)
                else:
                    _, preds = torch.max(outputs, 1)

                label_list.append(self.get_target_labels(labels).detach().cpu())
                pred_list.append(preds.detach().cpu())

        label_array = np.concatenate(label_list, axis=0)
        pred_array = np.concatenate(pred_list)

        return np.round(pred_array).astype(int), label_array, pred_array

    @abc.abstractmethod
    def per_class_accuracy(self, preds, labels):
        raise NotImplementedError(f"{self.__class__.__name__}, which is a subclass of BiasTrainer, should implement 'per_class_accuracy'")
