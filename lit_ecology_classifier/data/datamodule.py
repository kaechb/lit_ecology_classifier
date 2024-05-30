import logging
import os
from collections.abc import Iterable

import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split

from ..data.tardataset import TarImageDataset
from ..data.imagedataset import ImageFolderDataset
from ..helpers.helpers import TTA_collate_fn


class DataModule(LightningDataModule):
    """
    A LightningDataModule for handling image datasets stored in a tar file.
    This module is responsible for preparing and loading data in a way that is compatible
    with PyTorch training routines using the PyTorch Lightning framework.

    Attributes:
        tarpath (str): Path to the tar file containing the dataset.
        batch_size (int): Number of images to load per batch.
        dataset (str): Identifier for the dataset being used.
        testing (bool): Flag to enable testing mode, which includes TTA (Test Time Augmentation).
        use_multi (bool): Flag to enable multi-processing for data loading.
        priority_classes (str): Path to the JSON file containing a list of the priority classes.
        splits (Iterable): Proportions to split the dataset into training, validation, and testing.
    """

    def __init__(self, datapath: str, batch_size: int, dataset: str, TTA: bool = False, use_multi: bool = True, priority_classes: list = [], splits: Iterable = [0.7, 0.15], **kwargs):
        super().__init__()
        self.datapath = datapath
        self.TTA = TTA # Enable Test Time Augmentation if testing is True
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_multi = use_multi
        self.train_split, self.val_split = splits
        self.priority_classes = priority_classes
        self.class_map_path = f"./params/{dataset}/class_map.json"
        # Verify that class map exists for testing mode


    def setup(self, stage=None):
        """
        Prepares the datasets for training, validation, and testing by applying appropriate splits.
        This method also handles the TTA mode adjustments.

        Args:
            stage (Optional[str]): Current stage of the model training/testing. Not used explicitly in the method.
        """
        # Load the dataset
        if stage != "predict":
            if self.datapath.find(".tar") == -1:
                full_dataset = ImageFolderDataset(self.datapath, self.class_map_path, self.priority_classes, TTA=self.TTA, train=True)
            else:
                full_dataset = TarImageDataset(self.datapath, self.class_map_path, self.priority_classes, TTA=self.TTA, train=True)

            self.class_map_path = full_dataset.class_map_path
            self.class_map = full_dataset.class_map
            print("Number of classes:", len(self.class_map))

            # Calculate dataset splits
            train_size = int(self.train_split * len(full_dataset))
            val_size = int(self.val_split * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size

            # Randomly split the dataset into train, validation, and test sets
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
            # Set train flag to False for validation and test datasets
            self.val_dataset.train = False
            self.test_dataset.train = False
        else:

            self.predict_dataset = TarImageDataset(self.datapath, self.class_map_path, self.priority_classes, TTA=self.TTA, train=False)

            self.class_map = self.predict_dataset.class_map
            self.class_map_path = self.predict_dataset.class_map_path
        if not os.path.exists(self.class_map_path) and self.TTA:
            raise FileNotFoundError(f"Class map not found at {self.class_map_path}. The class map needs to exist for inference.")
    def train_dataloader(self):
        """
        Constructs the DataLoader for training data.
        Returns:
            DataLoader: DataLoader object for the training dataset.
        """
        # Use a distributed sampler if multiple GPUs are available and multi-processing is enabled
        sampler = DistributedSampler(self.train_dataset) if torch.cuda.device_count() > 1 and self.use_multi else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=os.cpu_count() // torch.cuda.device_count() if torch.cuda.device_count() > 0 else 4,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        Constructs the DataLoader for validation data.
        Returns:
            DataLoader: DataLoader object for the validation dataset.
        """
        sampler = DistributedSampler(self.val_dataset) if torch.cuda.device_count() > 1 and self.use_multi else None
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=os.cpu_count() // torch.cuda.device_count() if torch.cuda.device_count() > 0 else 4,
            pin_memory=True,
            drop_last=False,
        )
        if self.TTA:
            # Apply TTA collate function if TTA is enabled
            loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=os.cpu_count() // torch.cuda.device_count() if torch.cuda.device_count() > 0 else 4,
                pin_memory=True,
                drop_last=False,
                collate_fn=TTA_collate_fn,
            )
        return loader

    def test_dataloader(self):
        """
        Constructs the DataLoader for testing data.
        Returns:
            DataLoader: DataLoader object for the testing dataset.
        """
        sampler = DistributedSampler(self.test_dataset) if torch.cuda.device_count() > 1 and self.use_multi else None
        if self.TTA:
            loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=os.cpu_count() // torch.cuda.device_count() if torch.cuda.device_count() > 0 else 4,
                pin_memory=True,
                drop_last=False,
                collate_fn=TTA_collate_fn,
            )
        else:
            loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=os.cpu_count() // torch.cuda.device_count() if torch.cuda.device_count() > 0 else 4,
                pin_memory=True,
                drop_last=False,
            )
        return loader

    def predict_dataloader(self):
        """
        Constructs the DataLoader for inference on data.
        Returns:
            DataLoader: DataLoader object for the inference dataset.
        """
        sampler = DistributedSampler(self.predict_dataset) if torch.cuda.device_count() > 1 and self.use_multi else None
        if self.TTA:
            loader = DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=os.cpu_count() // torch.cuda.device_count() if torch.cuda.device_count() > 0 else 4,
                pin_memory=False,
                drop_last=False,
                collate_fn=TTA_collate_fn,
            )
        else:
            loader = DataLoader(
                self.predict_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=os.cpu_count() // torch.cuda.device_count() if torch.cuda.device_count() > 0 else 4,
                pin_memory=False,
                drop_last=False,
            )
        return loader



if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create an instance of the PlanktonDataModule with the specified parameters
    dm = DataModule(
        "/beegfs/desy/user/kaechben/eawag/training/phyto.tar", dataset="phyto", batch_size=1024, testing=False, use_multi=False, priority_classes="config/priority.json", splits=[0.7, 0.15]
    )
    # Set up datasets for the 'fit' stage
    dm.setup("fit")
    # Get a DataLoader for training and iterate through it
    test_loader = dm.train_dataloader()
    k = 0
    for i in test_loader:
        print(i[0].shape, len(i[1]))
        k += i[0].shape[0]
    print("number of images", k)
