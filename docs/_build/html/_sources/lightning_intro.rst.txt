Introduction to PyTorch Lightning
=================================

PyTorch Lightning is a high-level framework built on top of PyTorch, designed to make deep learning research and experimentation faster, more flexible, and more reproducible. This guide will help you understand what PyTorch Lightning is, why it is beneficial, and how you can extend it for your own projects.

What is PyTorch Lightning?
--------------------------

PyTorch Lightning provides a standardized interface for training deep learning models. It abstracts away much of the boilerplate code needed for PyTorch, allowing you to focus on the actual research and model development. Lightning helps streamline tasks such as training loops, validation, checkpointing, logging, and multi-GPU training.

Key Features of PyTorch Lightning:
----------------------------------

- **Standardized Structure**: Enforces a clean separation of model, data, and training code.
- **Automatic Checkpointing**: Saves model weights and other important information during training.
- **Logging**: Easily log metrics and visualize them using tools like TensorBoard.
- **Scalability**: Seamlessly scale your models to multiple GPUs or nodes with minimal code changes.
- **Flexibility**: Compatible with other PyTorch utilities and custom research code.

Why Use PyTorch Lightning?
--------------------------

1. **Ease of Use**: Simplifies complex training and validation loops, making your code more readable and maintainable.
2. **Reproducibility**: Standardized structure and logging make it easier to reproduce experiments and results.
3. **Scalability**: Built-in support for multi-GPU and distributed training allows you to scale up your experiments without rewriting your code.
4. **Focus on Research**: By abstracting away boilerplate code, Lightning allows you to focus more on the research and less on the implementation details.

Basic Components of PyTorch Lightning
-------------------------------------

1. **LightningModule**: This is where you define your model architecture, training, validation, and test steps.

   Example:
   .. code-block:: python

       import pytorch_lightning as pl
       import torch
       from torch import nn
       from torch.optim import Adam

       class LitModel(pl.LightningModule):
           def __init__(self):
               super(LitModel, self).__init__()
               self.layer = nn.Linear(28 * 28, 10)

           def forward(self, x):
               return self.layer(x.view(x.size(0), -1))

           def training_step(self, batch, batch_idx):
               x, y = batch
               y_hat = self(x)
               loss = nn.functional.cross_entropy(y_hat, y)
               return loss

           def configure_optimizers(self):
               return Adam(self.parameters(), lr=1e-3)

2. **LightningDataModule**: This is where you define your data loading and preprocessing logic. Here is an example of a `TarImageDataset` used within a `LightningDataModule`:

   .. code-block:: python

       import pytorch_lightning as pl
       from torch.utils.data import DataLoader
       from lit_ecology_classifier.data.datamodule import TarImageDataset

       class TarImageDataModule(pl.LightningDataModule):
           def __init__(self, tar_path: str, class_map_path: str, priority_classes: str, batch_size: int = 32, TTA: bool = False):
               super().__init__()
               self.tar_path = tar_path
               self.class_map_path = class_map_path
               self.priority_classes = priority_classes
               self.batch_size = batch_size
               self.TTA = TTA

           def setup(self, stage=None):
               self.train_dataset = TarImageDataset(self.tar_path, self.class_map_path, self.priority_classes, train=True, TTA=self.TTA)
               self.val_dataset = TarImageDataset(self.tar_path, self.class_map_path, self.priority_classes, train=False, TTA=self.TTA)

           def train_dataloader(self):
               return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

           def val_dataloader(self):
               return DataLoader(self.val_dataset, batch_size=self.batch_size)

How to Extend PyTorch Lightning
-------------------------------

1. **Custom Metrics**: You can define custom metrics for your training and validation steps by integrating with libraries like `torchmetrics`.

   Example:
   .. code-block:: python

       from torchmetrics import Accuracy

       class LitModel(pl.LightningModule):
           def __init__(self):
               super(LitModel, self).__init__()
               self.layer = nn.Linear(28 * 28, 10)
               self.accuracy = Accuracy()

           def training_step(self, batch, batch_idx):
               x, y = batch
               y_hat = self(x)
               loss = nn.functional.cross_entropy(y_hat, y)
               acc = self.accuracy(y_hat, y)
               self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
               return loss

2. **Custom Callbacks**: You can add custom callbacks to your training process to extend its functionality.

   Example:
   .. code-block:: python

       from pytorch_lightning.callbacks import Callback

       class CustomCallback(Callback):
           def on_train_start(self, trainer, pl_module):
               print("Training is starting!")

       trainer = pl.Trainer(callbacks=[CustomCallback()])

Conclusion
----------

PyTorch Lightning is a powerful tool that can simplify your deep learning workflows, improve reproducibility, and allow you to scale your experiments with minimal effort. By using Lightning, you can focus more on your research and less on the boilerplate code.

For more information, visit the `PyTorch Lightning Documentation <https://pytorch-lightning.readthedocs.io/en/latest/>`_.
