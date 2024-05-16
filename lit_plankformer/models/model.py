import torch
import numpy as np
from lightning import LightningModule

from torch.optim.lr_scheduler import OneCycleLR
from ..models.setup_model import setup_model
from ..helpers.helpers import gmean, output_results


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
        self.classes = np.load(self.hparams.main_param_path +self.hparams.dataset+ '/classes.npy')
        self.class_weights = torch.load(self.hparams.main_param_path+"/" +self.hparams.dataset+"/" + '/class_weights_tensor.pt').float()
        self.loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = OneCycleLR(optimizer, max_lr=self.hparams.lr, steps_per_epoch=len(self.datamodule.train_dataloader())-1, epochs=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def load_datamodule(self, datamodule):
        """
        Load the data module into the model.
        Args:
            datamodule (LightningDataModule): Data module to load.
        """
        self.datamodule = datamodule

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.
        Args:
            batch (tuple): Input batch containing images and labels.
            batch_idx (int): Batch index.
        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        sch = self.lr_schedulers() # this schedules the lr
        sch.step()
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss
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
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)


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
        pred_label = np.array(self.classes[max_index.cpu().numpy()], dtype=object)
        output_results(self.hparams.test_outpath,self.hparams.finetuned,self.filenames, pred_label)
        return super().on_test_epoch_end()

