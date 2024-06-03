import io
import json
import logging
import os
import pprint
import random
import tarfile
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2 import AugMix, Compose, Normalize, RandomHorizontalFlip, RandomRotation, Resize, ToDtype, ToImage

from ..helpers.helpers import define_priority_classes, define_rest_classes


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

    def __init__(self, tar_path: str, class_map_path: str, priority_classes: list, rest_classes:list,train: bool, TTA: bool = False):
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
        self.train = train
        self.class_map_path = class_map_path
        self.priority_classes = priority_classes
        self.rest_classes = rest_classes

        # Load priority classes and adjust class map accordingly
        if self.priority_classes != []:

            logging.info(f"Priority classes not None. Loading priority classes from {self.priority_classes}")

            priority_postfix = "_priority"
            logging.info(f"Priority classes loaded: {self.priority_classes}")
            self.class_map_path = self.class_map_path.replace("class_map.json", f"class_map{priority_postfix}.json")
            logging.info(f"Class map path set to {self.class_map_path}")

        elif self.rest_classes != []:

            logging.info(f"rest classes not None. Loading rest classes from {self.rest_classes}")

            rest_postfix = "_rest"
            logging.info(f"rest classes loaded: {self.rest_classes}")
            self.class_map_path = self.class_map_path.replace("class_map.json", f"class_map{rest_postfix}.json")
            logging.info(f"Class map path set to {self.class_map_path}")

        # Load class map from JSON or extract it from the tar file if not present
        if not os.path.exists(self.class_map_path):
            if not train:
                raise FileNotFoundError(f"Class map not found at {self.class_map_path}. Class map needs to be present for testing.")
            logging.info(f"Class map not found at {self.class_map_path}. Extracting class map from tar file.")
            self._extract_class_map(tar_path)
            logging.info(f"Class map saved to {self.class_map_path}")
        else:
            logging.info(f"Loading class map from {self.class_map_path}")
            with open(self.class_map_path, "r") as json_file:
                self.class_map = json.load(json_file)
            logging.info(f"Class map loaded.")

        # Transformation sequences for training and validation/testing
        self._define_transforms()
        # Load image information from the tar file
        self.image_infos = self._load_image_infos()
        if self.rest_classes:
            self._filter_rest_classes()


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
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # ImageNet mean and std
        self.train_transforms = Compose([ToImage(), RandomHorizontalFlip(), Resize((224, 224)), ToDtype(torch.float32, scale=True), AugMix(), Normalize(mean, std)])
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
            label = self.get_label_from_filename(image_info.name)
            return image, label

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

    def _extract_class_map(self, tar_path):
        """
        Extracts the class map from the contents of the tar file and saves it to a JSON file.
        """
        logging.info("Extracting class map from tar file.")
        class_map = {}

        with tarfile.open(tar_path, "r") as tar:
            # Temporary set to track folders that contain images
            folders_with_images = set()

            # First pass: Identify folders containing images
            for member in tar.getmembers():
                if member.isdir():
                    continue  # Skip directories
                if member.isfile() and member.name.lower().endswith(("jpg", "jpeg", "png")):
                    class_name = os.path.basename(os.path.dirname(member.name))
                    folders_with_images.add(class_name)

            # Second pass: Build the class map only for folders with images
            for member in tar.getmembers():
                if member.isdir():
                    continue  # Skip directories
                class_name = os.path.basename(os.path.dirname(member.name))
                if class_name in folders_with_images:
                    if class_name not in class_map:
                        class_map[class_name] = []
                    class_map[class_name].append(member.name)

        # Create a sorted list of class names and map them to indices
        sorted_class_names = sorted(class_map.keys())
        logging.info(f"Found {len(sorted_class_names)} classes.")
        self.class_map = {class_name: idx for idx, class_name in enumerate(sorted_class_names)}
        if self.priority_classes != []:

            logging.info(f'priority_classes not set to []. Defining priority class_map')
            for key in self.priority_classes:
                if key not in self.class_map.keys():
                    raise KeyError(f"Priority class {key} not found in class map. Keys of class map: {pprint.pformat(self.class_map.keys())}")
            self.class_map = define_priority_classes(self.priority_classes)
        if self.rest_classes != []:

            logging.info(f'rest_classes not set to []. Defining rest class_map')
            for key in self.rest_classes:
                if key not in self.class_map.keys():
                    raise KeyError(f"rest class {key} not found in class map. Keys of class map: {pprint.pformat(self.class_map.keys())}")
            self.class_map = define_rest_classes(self.rest_classes)

        logging.info(f"Class map created:\n{pprint.pformat(self.class_map)}")
        logging.info(f"Saving class map to {self.class_map_path}")
        os.makedirs(os.path.dirname(self.class_map_path), exist_ok=True)
        with open(self.class_map_path, "w") as json_file:
            json.dump(self.class_map, json_file, indent=4)

    def get_label_from_filename(self, filename):
        """
        Extracts the label index from a given filename.

        Args:
            filename (str): The filename from which to extract the label.

        Returns:
            int: The label index corresponding to the class.
        """
        label = filename.split("/")[1]
        if self.priority_classes != []:
            label = self.class_map.get(label, 0)
        else:
            label = self.class_map[label]
        return label

    def shuffle(self):
        """
        Shuffles the list of image information to randomize data access, useful during training.
        """
        random.shuffle(self.image_infos)
