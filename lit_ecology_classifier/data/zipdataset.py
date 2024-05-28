import io
import json
import logging
import os
import zipfile
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import AugMix, Compose, Normalize, RandomHorizontalFlip, RandomRotation, Resize, ToDtype, ToImage


class ZipImageDataset(Dataset):
    """
    A Dataset subclass for managing and accessing image data stored in zip files. This class supports optional
    image transformations, and Test Time Augmentation (TTA) for enhancing model evaluation during testing.

    Attributes:
        zip_path (str): Path to the zip file containing image data.
        class_map_path (str): Path to the JSON file mapping class names to labels.
        priority_classes (str): Path to a JSON file specifying priority classes for targeted training or evaluation.
        train (bool): Specifies whether the dataset will be used for training. Determines the type of transformations applied.
        TTA (bool): Indicates if Test Time Augmentation should be applied during testing.
    """

    def __init__(self, zip_path: str, class_map_path: str, priority_classes: str, train: bool, TTA: bool = False):
        """
        Initializes the ZipImageDataset with paths and modes.
        """
        self.zip_path = zip_path
        self.TTA = TTA
        self.train = train
        self.class_map_path = class_map_path
        self.priority_classes = priority_classes
        self.zip_file = zipfile.ZipFile(zip_path, "r")
        self.file_list = []
        self.labels = []
        self.class_map = {}

        logging.basicConfig(level=logging.INFO)

        # Load priority classes and adjust class map accordingly
        if os.path.exists(self.priority_classes):
            with open(self.priority_classes, "r") as json_file:
                priority_classes_data = json.load(json_file)
                logging.info(f"Priority classes loaded: {priority_classes_data['priority_classes']}")
        else:
            logging.warning("Priority classes file not found.")

        if not os.path.exists(self.class_map_path):
            logging.info("Class map not found. Extracting class map from zip file.")
            self._extract_class_map()
            logging.info(f"Class map saved to {self.class_map_path}")
        else:
            logging.info(f"Loading class map from {self.class_map_path}")
            with open(self.class_map_path, "r") as file:
                self.class_map = json.load(file)
            logging.info("Class map loaded.")

        # Load images and labels
        self._load_image_infos()

        # Define transformations
        self._define_transforms()

    def _define_transforms(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet mean and std
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
        return len(self.file_list)

    def __getitem__(self, idx):
        file_info = self.file_list[idx]
        image_data = self.zip_file.read(file_info["filename"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        if self.TTA:
            image = {rot: self.val_transforms(self.rotations[rot](image)) for rot in self.rotations}
        elif self.train:
            image = self.train_transforms(image)
        else:
            image = self.val_transforms(image)

        label = self.labels[idx]
        return image, label

    def _load_image_infos(self):
        for filename in self.zip_file.namelist():
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                class_name = os.path.dirname(filename).split("/")[-1]
                label = self.class_map.get(class_name)
                if label is not None:
                    self.file_list.append({"filename": filename, "label": label})
                    self.labels.append(label)

    def _extract_class_map(self):
        class_names = set()
        for filename in self.zip_file.namelist():
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                class_name = os.path.dirname(filename).split("/")[-1]
                class_names.add(class_name)
        self.class_map = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}

        with open(self.class_map_path, "w") as file:
            json.dump(self.class_map, file, indent=4)
        logging.info(f"Class map created and saved to {self.class_map_path}")

    def close(self):
        self.zip_file.close()
        logging.info("Zip file closed.")
