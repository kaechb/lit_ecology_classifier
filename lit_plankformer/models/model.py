import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score, f1_score
from ..helpers.helpers import (CosineWarmupScheduler, FocalLoss, gmean,
                               output_results, plot_confusion_matrix,
                               plot_score_distributions)
from ..models.setup_model import setup_model


class Plankformer(LightningModule):
    def __init__(self, **hparams):
        """
        Initialize the Plankformer model.
        Args:
            hparams (dict): Hyperparameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = setup_model(**self.hparams)

        self.class_weights = torch.load(self.hparams.main_param_path+"/" +self.hparams.dataset+"/" + '/class_weights.pt').float()
        print("hparams",list(self.hparams))
        self.loss = torch.nn.CrossEntropyLoss(weight=self.class_weights if self.hparams.balance_weight else None) if not "loss" in list(self.hparams) or not self.hparams.loss=="focal" else FocalLoss(alpha=self.class_weights if self.hparams.balance_weight else None,gamma=self.hparams.gamma)

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
        if self.hparams.use_scheduler:
            print("max_iters:",self.trainer.max_epochs*len(self.datamodule.train_dataloader()))
            scheduler = CosineWarmupScheduler(optimizer, warmup=3*len(self.datamodule.train_dataloader()), max_iters=self.trainer.max_epochs*len(self.datamodule.train_dataloader()))
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [lr_scheduler_config]
        else:
            return optimizer

    def load_datamodule(self, datamodule):
        """
        Load the data module into the model.
        Args:
            datamodule (LightningDataModule): Data module to load.
        """
        self.datamodule = datamodule
        self.class_map = self.datamodule.class_map
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
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,sync_dist=True)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_step_outputs = []
        self.val_step_targets = []
        self.val_step_logits = []
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
            print(batch)
            x=torch.cat([batch[0][str(i*90)] for i in range(4)],dim=0)
            y = batch[1]
            logits = self(x).softmax(dim=1)
            logits = torch.stack(torch.chunk(logits, 4, dim=0))
            logits=gmean(logits,dim=0)
        else:
            x, y = batch
            logits = self(x)

        loss = self.loss(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        f1 = f1_score(y.cpu(), logits.argmax(dim=1).cpu(), average='weighted')
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        self.val_step_logits.append(logits.softmax(dim=1).cpu())
        self.val_step_outputs.append(logits.cpu().softmax(dim=1).argmax(dim=1))
        self.val_step_targets.append(y.cpu())

        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1, 'logits': logits, 'y': y}

    def on_validation_epoch_end(self):
        """
        Aggregate outputs and log the confusion matrix at the end of the validation epoch.
        Args:
            outputs (list): List of dictionaries returned by validation_step.
        """
        all_scores = torch.cat(self.val_step_logits)
        all_preds = torch.cat(self.val_step_outputs)
        all_labels = torch.cat(self.val_step_targets)
        fig_score=plot_score_distributions(all_scores, all_preds, self.inverted_class_map.values(),all_labels)
        balanced_acc= balanced_accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
        self.log('val_balanced_acc', balanced_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        false_positives = torch.sum((all_labels == 0) & (all_preds != 0))/torch.sum(all_labels == 0)
        self.log('val_false_positives', false_positives.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        # Log the confusion matrix to wandb if use_wandb is true
        if self.hparams.use_wandb:


            fig,fig2=plot_confusion_matrix(all_labels, all_preds,self.inverted_class_map.values())
            plt.close(fig)
            plt.close(fig2)


            self.logger.log_image(key=f"score_distributions",images=[fig_score],step=self.current_epoch)

            self.logger.log_image(key="confusion_matrix",images=[fig],step=self.current_epoch)
            self.logger.log_image(key="confusion_matrix_norm",images=[fig2],step=self.current_epoch)
        else:

            fig_score.savefig(f"score_distributions.png")

    def on_test_epoch_start(self) -> None:
        """
        Hook to be called at the start of the test epoch.
        """
        self.probabilities = []
        self.filenames = []


        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        """
        Perform a test step.
        Args:
            batch (tuple): Input batch containing images and filenames.
            batch_idx (int): Batch index.
        """
        with torch.no_grad():
            probs = []

            if self.hparams.TTA:
                images, filenames = zip(*batch.values())
                images = torch.cat(images, dim=0)  # Combine all images into a single tensor
                filenames = filenames[0]  # Assume filenames are the same for all augmentations
            else:
                images, filenames = batch

            for m in self.model:
                m.to(self.device)
                m.eval()
                logits = m(images.to(self.device)).cpu()

                if self.hparams.TTA:
                    logits = torch.chunk(logits, 4, dim=0)
                    # Average predictions from the different augmentations
                    probs.extend([torch.nn.functional.softmax(log, dim=1) for log in logits])
                else:
                    probs.append(torch.nn.functional.softmax(logits, dim=1))

            self.filenames.extend(filenames)
            self.probabilities.append(gmean(torch.stack(probs, dim=-1), dim=-1))

    def on_test_epoch_end(self) -> None:
        """
        Hook to be called at the end of the test epoch.
        """
        max_index = torch.cat(self.probabilities).argmax(axis=1)
        pred_label = np.array(self.inverted_class_map[max_index.cpu().numpy()], dtype=object)
        output_results(self.hparams.test_outpath,self.filenames, pred_label)
        return super().on_test_epoch_end()

