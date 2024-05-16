import io
import random
import tarfile

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import Compose, RandomRotation, Resize, ToPILImage, ToTensor, Lambda, Pad

from ..helpers.load_images import load_images
from ..helpers.image_edits import get_padding

class TarImageDataset(Dataset):
    def __init__(self, tar_path,dataset="phyto",use_data_moments=True):
        self.tar_path = tar_path
        if use_data_moments:
            mean, std= [-1.8272, -1.7527, -1.5727], [0.7074, 0.6921, 0.5333] # calculated from the phyto dataset - see calc_normalisation.py
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # imagenet normalization
        self.transform = transforms.Compose([
            Lambda(lambda img: Pad(get_padding(img), fill=0, padding_mode='constant')(img)),  # Add dynamic padding
            transforms.Resize((224, 224)),                 # Resize shortest side to 256, preserving aspect ratio
            transforms.RandomHorizontalFlip(),             # Randomly flip the image horizontally
            transforms.ToTensor(),                         # Convert the image to a tensor
            transforms.Normalize(mean, std)  # Normalize the image #TODO calculate normalization
        ])
        self.image_infos = self._load_image_infos()

        self.class_map = np.load(f'./params/{dataset}/classes.npy', allow_pickle=True)
        self.class_map = {class_name: idx for idx, class_name in enumerate(self.class_map)}

    def _load_image_infos(self):
        image_infos = []
        with tarfile.open(self.tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(('jpg', 'jpeg', 'png')):
                    image_infos.append(member)
        return image_infos

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        with tarfile.open(self.tar_path, 'r') as tar:
            image_info = self.image_infos[idx]
            image_file = tar.extractfile(image_info)
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

            if self.transform:
                image = self.transform(image)

            # Assuming the labels are part of the filename, you can modify this according to your needs
            label = self._get_label_from_filename(image_info.name)

            return image, label

    def _get_label_from_filename(self, filename):
        # Custom function to extract label from filename
        # Assuming labels are encoded in the filename, e.g., "cat.1234.jpg"
        label = filename.split('/')[1]
        label = self.class_map[label]
        return label

    def shuffle(self):
        random.shuffle(self.image_infos)

class CreatePytorchDataset(Dataset):
    """
        Dataset that includes a rotation by 0,90,180,270 degrees depending on the value of rot
        This is used for testing, where we want to apply test-time augmentation
    """
    def __init__(self, X,filenames, rot=0):
        self.X = X
        self.transform = Compose([
            ToPILImage(),
            Resize(224),
            RandomRotation(degrees=(90*rot, 90*rot)),
            ToTensor()
        ])
        self.filenames = filenames
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        filenames = self.filenames[index]
        return self.transform(image),filenames

class PlanktonDataModule(LightningDataModule):
    def __init__(self, datapath, L=128, resize_images=None, TTA=True, batch_size=32, dataset=""):
        super().__init__()
        self.datapath = datapath
        self.L = L
        self.resize_images = resize_images
        self.TTA = TTA
        self.batch_size = batch_size
        self.classes = np.load(f'./params/{dataset}/classes.npy')
        self.class_weights_tensor = torch.load(f'./params/{dataset}/class_weights_tensor.pt')
        self.dataset=dataset

    def setup(self, stage=None):
        # Load and prepare data

        if stage == 'fit':
            full_dataset = TarImageDataset(self.datapath,dataset=self.dataset)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        elif stage == 'test' or stage is None:
            df = load_images(self.datapath, self.L, self.resize_images)
            self.filenames = df.filename
            # Basic checks on the dataset
            self.check_format(df)
            X_np = np.stack(df['npimage'].values).astype(np.float64)
            X_np = (255 * X_np).astype(np.uint8)
            if self.TTA:
                self.test_dataset = [CreatePytorchDataset(X_np, rot=i,filenames=df.filename) for i in range(0,4)]
            else:
                self.test_dataset = CreatePytorchDataset(X_np,df.filename)

    def check_format(self, df):
        if df.isnull().any().any():
            raise ValueError("There are NaN values in the data.")
        if 'npimage' in df.columns and df.npimage.iloc[0].shape != (self.L, self.L, 3):
            raise ValueError("Images have the incorrect shape")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    def test_dataloader(self):

        if self.TTA:
            loader = CombinedLoader({rot : DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,drop_last=False) for rot,ds in zip(["0","90","180","270"],self.test_dataset)})
        else:
            loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,drop_last=False)
        return loader

if __name__ == '__main__':
    import os
    dm = ZooplanktonDataModule('/scratch/snx3000/bkch/training/Phytolake1.tar',dataset="phyto")
    dm.setup('fit')
    test_loader = dm.train_dataloader()
    k=0
    for i in test_loader:
        print(i[0].shape,len(i[1]))
        k+=i[0].shape[0]
    print("number of images",k)