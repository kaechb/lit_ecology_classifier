import logging
import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score, f1_score

from ..helpers.helpers import CosineWarmupScheduler, gmean, output_results, plot_confusion_matrix, plot_loss_acc, plot_score_distributions
from ..models.setup_model import setup_model


class LitClassifier(LightningModule):
    def __init__(self, **hparams):
        """
        Initialize the LitClassifier.
        Args:
            hparams (dict): Hyperparameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = setup_model(**self.hparams)
        if self.hparams.class_weights is not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.hparams.class_weights, dtype=torch.float32))
        else:
            self.loss = torch.nn.CrossEntropyLoss()
        logging.info("Model initialized with hyperparameters:\n {}".format(pprint.pformat(self.hparams)))

    def TTA(self, batch):
        """
        Perform Test Time Augmentation (TTA) on the input batch.
        Args:
            batch (tuple): Input batch containing images and labels.
        Returns:
            torch.Tensor: Geometrics Average of probabilities from the TTA predictions.
            torch.Tensor: True labels if batch is list containg true labels as second entry else None.
        """


        x = torch.cat([batch[0][str(i * 90)] for i in range(4)], dim=0)
        logits = self(x).softmax(dim=1)
        logits = torch.stack(torch.chunk(logits, 4, dim=0))
        logits = gmean(logits, dim=0)
        return logits

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Model output.
        """
        return self.model(x)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        Returns:
            list: List of optimizers.
            list: List of schedulers.
        """

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

        scheduler = CosineWarmupScheduler(optimizer, warmup=3 * len(self.datamodule.train_dataloader()), max_iters=self.trainer.max_epochs * len(self.datamodule.train_dataloader()))
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]

    def load_datamodule(self, datamodule):
        """
        Load the data module into the model.
        Args:
            datamodule (LightningDataModule): Data module to load.
        """
        self.datamodule = datamodule
        self.class_map = self.datamodule.class_map
        self.hparams.TTA = self.datamodule.TTA
        self.inverted_class_map = dict(sorted({v: k for k, v in self.class_map.items()}.items()))

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.
        Args:
            batch (tuple): Input batch containing images and labels.
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_step_predictions = []
        self.val_step_targets = []
        self.val_step_probs = []

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step.
        Args:
            batch (tuple): Input batch containing images and labels.
            batch_idx (int): Batch index.
        Returns:
            dict: Dictionary containing the loss and predictions.
        """

        x, y = batch
        logits = self(x)

        loss = self.loss(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        f1 = f1_score(y.cpu(), logits.argmax(dim=1).cpu(), average="weighted")
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.val_step_probs.append(logits.softmax(dim=1).cpu())
        self.val_step_predictions.append(logits.cpu().argmax(dim=1))
        self.val_step_targets.append(y.cpu())

        return {"val_loss": loss, "val_acc": acc, "val_f1": f1, "logits": logits, "y": y}

    def on_validation_epoch_end(self):
        """
        Aggregate outputs and log the confusion matrix at the end of the validation epoch.
        Args:
            outputs (list): List of dictionaries returned by validation_step.
        """
        all_scores = torch.cat(self.val_step_probs)
        all_preds = torch.cat(self.val_step_predictions)
        all_labels = torch.cat(self.val_step_targets)
        fig_score = plot_score_distributions(all_scores, all_preds, self.inverted_class_map.values(), all_labels)
        balanced_acc = balanced_accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
        self.log("val_balanced_acc", balanced_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        false_positives = torch.sum((all_labels == 0) & (all_preds != 0)) / torch.sum(all_labels == 0)
        self.log("val_false_positives", false_positives.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        fig, fig2 = plot_confusion_matrix(all_labels, all_preds, self.inverted_class_map.values())
        # Log the confusion matrix to wandb if use_wandb is true
        if self.hparams.use_wandb:

            self.logger.log_image(key=f"score_distributions", images=[fig_score], step=self.current_epoch)
            self.logger.log_image(key="confusion_matrix", images=[fig], step=self.current_epoch)
            self.logger.log_image(key="confusion_matrix_norm", images=[fig2], step=self.current_epoch)
        else:
            fig.savefig(f"{self.hparams.train_outpath}/confusion_matrix_epoch_{self.current_epoch}.png")
            fig2.savefig(f"{self.hparams.train_outpath}/confusion_matrix_normalized_epoch_{self.current_epoch}.png")
            fig_score.savefig(f"{self.hparams.train_outpath}/score_distributions_epoch_{self.current_epoch}.png")
        plt.close(fig)
        plt.close(fig2)
        plt.close(fig_score)

    def on_test_epoch_start(self) -> None:
        """
        Hook to be called at the start of the test epoch.
        Sets up empty lists to store the predicted class probabilities and filenames.
        """
        self.test_step_predictions = []
        self.test_step_targets = []
        self.test_step_probs = []
        self.model.eval()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        """
        Perform a test step.
        Args:
            batch (tuple): Input batch containing images and filenames.
            batch_idx (int): Batch index.
        """
        with torch.no_grad():
            if self.hparams.TTA:
                probs = self.TTA(batch)
                y=batch[1]
            else:
                x,y = batch
                probs = self(x).softmax(dim=1).cpu()
            self.test_step_targets.append(y.cpu())
            self.test_step_predictions.append(probs.argmax(1).cpu())
            self.test_step_probs.append(probs.cpu())

    def on_test_epoch_end(self):
        """
        Aggregate outputs and log the confusion matrix at the end of the test epoch.
        Args:
            outputs (list): List of dictionaries returned by test_step.
        """
        all_scores = torch.cat(self.test_step_probs)
        all_preds = torch.cat(self.test_step_predictions)
        all_labels = torch.cat(self.test_step_targets)
        fig_score = plot_score_distributions(all_scores, all_preds, self.inverted_class_map.values(), all_labels)
        balanced_acc = balanced_accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
        self.log("test_balanced_acc", balanced_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        false_positives = torch.sum((all_labels == 0) & (all_preds != 0)) / torch.sum(all_labels == 0)
        self.log("test_false_positives", false_positives.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        fig, fig2 = plot_confusion_matrix(all_labels, all_preds, self.inverted_class_map.values())

        if self.hparams.use_wandb:
            self.logger.log_image(key=f"test_score_distributions", images=[fig_score], step=self.current_epoch)
            self.logger.log_image(key="test_confusion_matrix", images=[fig], step=self.current_epoch)
            self.logger.log_image(key="test_confusion_matrix_norm", images=[fig2], step=self.current_epoch)
        else:
            logging.info(f"Saving confusion matrix and score distributions to {self.hparams.outpath}")
            fig.savefig(f"{self.hparams.outpath}/test_confusion_matrix_test_set.png")
            fig2.savefig(f"{self.hparams.outpath}/test_confusion_matrix_normalized_test_set.png")
            fig_score.savefig(f"{self.hparams.outpath}/test_score_distributions_epoch_test_set.png")
        plt.close(fig)
        plt.close(fig2)
        plt.close(fig_score)

    def on_predict_start(self) -> None:
        """
        Hook for the start of the inference phase.
        """

        self.probabilities = []
        self.model.eval()

        return super().on_predict_start()

    def predict_step(self, batch) -> None:
        """
        Perform a prediction step on unlabeled data.
        Args:
            batch (tuple): Input batch containing images
        """
        with torch.no_grad():

            if self.hparams.TTA:
                probs = self.TTA(batch).cpu()
            else:
                batch = batch
                probs = self(batch).softmax(dim=1).cpu()
            self.probabilities.append(probs)

    def on_predict_epoch_end(self) -> None:
        """
        Hook to be called at the end of the test epoch.
        Saves predicted labels in text file in folder Output
        """
        filenames = self.datamodule.predict_dataset.image_infos
        max_index = torch.cat(self.probabilities).argmax(axis=1)

        pred_label = np.array([self.inverted_class_map[idx] for idx in max_index.numpy()], dtype=object)
        pred_score = torch.cat(self.probabilities).max(1)[0].numpy()
        output_results(self.hparams.outpath, filenames, pred_label, pred_score)
        plt.hist(max_index.numpy(), bins=len(self.inverted_class_map))
        plt.savefig(f"{self.hparams.outpath}/predictions_histogram.png")
        return super().on_test_epoch_end()

    def on_fit_end(self) -> None:
        """
        If the model is not using wandb, plot the loss and accuracy curves at the end of training
        and save them in the output folder.
        """
        if not self.hparams.use_wandb:
            plot_loss_acc(self.trainer.logger)
        return super().on_fit_end()
