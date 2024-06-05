import io
import json
import logging
import os
import pprint
import random
import tarfile
from collections import defaultdict
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2 import AugMix, Compose, Normalize, RandomHorizontalFlip, RandomRotation, Resize, ToDtype, ToImage

from ..helpers.helpers import define_priority_classes, define_rest_classes
from typing import Any


class TarImageDataset(Dataset):
    """
    A Dataset subclass for managing and accessing image data stored in tar files. This class supports optional
    image transformations, and Test Time Augmentation (TTA) for enhancing model evaluation during testing.

    Attributes:
        tar_path (str): Path to the tar file containing image data.
        class_map_path (str): Path to the JSON file mapping class names to labels.
        priority_classes (str): Path to a JSON file specifying priority classes for targeted training or evaluation.
        train (bool): Specifies whether the dataset will be used for training. Determines the type of transformations applied.
        TTA (bool): Indicates if Test Time Augmentation should be applied during testing.
    """

    def __init__(self, tar_path: str,class_map: dict, priority_classes:list, rest_classes:list, TTA: bool = False, train: bool = False):
        """
        Initializes the TarImageDataset with paths and modes.

        Args:
            tar_path (str): The file path to the tar archive containing the images.
            class_map_path (str): The file path to the JSON file with class mappings.
            priority_classes (str): The file path to the JSON file that contains priority classes.
            train (bool): A flag to indicate if the dataset is used for training purposes.
            TTA (bool): A flag to enable Test Time Augmentation.
        """
        self.tar_path = tar_path
        self.TTA = TTA
        self.class_map = class_map
        self.train = train
        self.priority_classes = priority_classes
        self.rest_classes = rest_classes
        # Transformation sequences for training and validation/testing
        self._define_transforms()
        # Load image information from the tar file
        self.image_infos = self._load_image_infos()
        if rest_classes!=[] and train:
            self._filter_rest_classes()
        self.train=train




    def _filter_rest_classes(self):
        """
        Removes samples that are not in rest_classes from the dataset.
        """
        logging.info(f"Filtering dataset to keep only classes in {self.rest_classes}")
        filtered_image_infos = []
        for image_info in self.image_infos:
            class_name = os.path.basename(os.path.dirname(image_info.name))
            if class_name in self.rest_classes:
                filtered_image_infos.append(image_info)
        self.image_infos = filtered_image_infos
        logging.info(f"Filtered dataset to {len(self.image_infos)} samples.")

    def _define_transforms(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet mean and std
        self.train_transforms = Compose([ToImage(), RandomHorizontalFlip(), RandomRotation(180), AugMix(), Resize((224, 224)), ToDtype(torch.float32, scale=True), Normalize(mean, std)])
        self.val_transforms = Compose([ToImage(), Resize((224, 224)), ToDtype(torch.float32, scale=True), Normalize(mean, std)])
        if self.TTA:
            self.rotations = {
                "0": Compose([RandomRotation(0, 0)]),
                "90": Compose([RandomRotation((90, 90))]),
                "180": Compose([RandomRotation((180, 180))]),
                "270": Compose([RandomRotation((270, 270))]),
            }

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total number of images.
        """
        return len(self.image_infos)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label based on the provided index.

        Args:
            idx (int): The index of the image.

        Returns:
            tuple: A tuple containing the transformed image and its label.
        """
        with tarfile.open(self.tar_path, "r") as tar:
            image_info = self.image_infos[idx]
            image_file = tar.extractfile(image_info)
            image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
            # Apply TTA transformations if enabled
            if self.TTA:
                image = {rot: self.val_transforms(self.rotations[rot](image)) for rot in self.rotations}
            elif self.train:
                image = self.train_transforms(image)
            else:
                image = self.val_transforms(image)
            if self.train:
                label = self.get_label_from_filename(image_info.name)
                return image, label
            else:
                return image

    def _load_image_infos(self):
        """
        Load image information from the tar file.
        """
        image_infos = []
        with tarfile.open(self.tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(("jpg", "jpeg", "png")):
                    image_infos.append(member)
        return image_infos



    def get_label_from_filename(self, filename):
        """
        Extracts the label index from a given filename.

        Args:
            filename (str): The filename from which to extract the label.

        Returns:
            int: The label index corresponding to the class.
        """

        label = filename.split("/")[1]
        if self.priority_classes!=[]:
            label = self.class_map.get(label, 0)
        else:
            label = self.class_map[label]
        return label

    def shuffle(self):
        """
        Shuffles the list of image information to randomize data access, useful during training.
        """
        random.shuffle(self.image_infos)
