import logging
import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

from ..helpers.helpers import CosineWarmupScheduler, gmean, output_results, plot_confusion_matrix, plot_loss_acc, plot_score_distributions, FocalLoss, setup_classmap, compute_roc_auc, compute_macro_precision_recall, compute_roc_auc_binary
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

        if 'class_map' not in self.hparams:
            self.hparams.class_map = setup_classmap(datapath=self.hparams['datapath'], priority_classes=self.hparams['priority_classes'], rest_classes=self.hparams['rest_classes'])
            self.class_map = self.hparams.class_map
            self.hparams.num_classes = len(self.class_map.keys())
        else:
            self.class_map = self.hparams.class_map
        self.inverted_class_map = dict(sorted({v: k for k, v in self.class_map.items()}.items()))
        self.model = setup_model(**self.hparams)
        self.loss = torch.nn.CrossEntropyLoss() if not "loss" in list(self.hparams) or not self.hparams.loss=="focal" else FocalLoss(alpha=None ,gamma=1.75)
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


        x = torch.cat([batch[str(i * 90)] for i in range(4)], dim=0)
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
        self.hparams.TTA = self.datamodule.TTA


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
        if self.hparams.TTA:
            probs = self.TTA(batch[0])
            logits=probs
            y=batch[1]
        else:
            x, y = batch
            logits = self(x)
            probs=logits.softmax(dim=1)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        acc = (probs.argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        f1 = f1_score(y.cpu(), probs.argmax(dim=1).cpu(), average="weighted")
        self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.val_step_probs.append(probs.cpu())
        self.val_step_predictions.append(probs.cpu().argmax(dim=1))
        self.val_step_targets.append(y.cpu())

        return {"val_loss": loss, "val_acc": acc, "val_f1": f1, "probs": probs, "y": y}

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
        precision = torch.sum((all_preds!= 0) & (all_labels!=0) ).item()/max(torch.sum((all_preds!= 0) & (all_labels!=0) ).item()+torch.sum((all_preds != 0) & (all_labels == 0)).item(),1)
        self.log("val_precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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

    def compute_metrics(self, all_labels, all_preds):
        # Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(
            all_labels.cpu().numpy(), all_preds.cpu().numpy()
        )
        # Calculate false positive rate
        false_positives = torch.sum((all_labels == 0) & (all_preds != 0)) / torch.sum(
            all_labels == 0
        )
        return balanced_acc, false_positives.item()



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
                probs = self.TTA(batch[0])
                y=batch[1]
            else:
                x,y = batch
                logits = self(x)
                probs=logits.softmax(dim=1)
            self.test_step_targets.append(y.cpu())
            self.test_step_predictions.append(probs.argmax(1).cpu())
            self.test_step_probs.append(probs.cpu())

    def on_test_epoch_end(self):
        """
        Aggregate outputs and log metrics and plots at the end of the test epoch.
        """
        import matplotlib.pyplot as plt

        # Aggregate outputs
        all_scores = torch.cat(self.test_step_probs)  # Shape: (N_samples, N_classes)
        all_preds = torch.cat(self.test_step_predictions)
        all_labels = torch.cat(self.test_step_targets)
        class_names = list(self.inverted_class_map.values())

        # Plot score distributions
        fig_score = plot_score_distributions(
            all_scores, all_preds, class_names, all_labels
        )

        # Compute metrics
        balanced_acc, false_positives = self.compute_metrics(all_labels, all_preds)


        # Plot confusion matrices
        plot_confusion_matrix(
            all_labels, all_preds, class_names, self.hparams.use_wandb, self.hparams.outpath


        )

        # Compute ROC curves and AUC
        roc_auc = compute_roc_auc(all_labels, all_scores)



        roc_auc_binary = compute_roc_auc_binary(all_labels, all_scores)
        # Compute F1 scores per class


        precision, recall, f1 = compute_macro_precision_recall(all_labels, all_preds)


        print("test_roc_auc_binary:", roc_auc_binary)
        print("test_balanced_acc:", balanced_acc)
        print("test_false_positives:", false_positives)
        print("test_acc:", accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy()))
        print("test_f1:", f1)
        print("test_auc_macro:", roc_auc)
        print("test_precision:", precision)
        print("test_recall:", recall)



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
        output_results(self.hparams.outpath, filenames, pred_label, pred_score, priority_classes=self.hparams.priority_classes!=[], rest_classes=self.hparams.rest_classes!=[], tar_file=self.hparams.datapath.find(".tar") != -1)
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
