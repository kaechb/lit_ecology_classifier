from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torchvision.transforms import Compose, ToPILImage, Resize, RandomRotation, ToTensor
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from helpers.load_images import load_images

class CreatePytorchDataset(Dataset):
    """Dataset that includes a rotation by 0,90,180,270 degrees depending on the value of rot"""
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

class ZooplanktonDataModule(LightningDataModule):
    def __init__(self, datapath, L=128, resize_images=None, TTA=True, batch_size=32):
        super().__init__()
        self.datapath = datapath
        self.L = L
        self.resize_images = resize_images
        self.TTA = TTA
        self.batch_size = batch_size
        self.classes = np.load('./data/classes.npy')
        self.class_weights_tensor = torch.load('./data/class_weights_tensor.pt')

    def setup(self, stage=None):
        # Load and prepare data
        df = load_images(self.datapath, self.L, self.resize_images)
        self.filenames = df.filename
        # Basic checks on the dataset
        self.check_format(df)
        X_np = np.stack(df['npimage'].values).astype(np.float64)
        X_np = (255 * X_np).astype(np.uint8)

        if stage == 'test' or stage is None:
            if self.TTA:
                self.test_dataset = [CreatePytorchDataset(X_np, rot=i,filenames=df.filename) for i in range(0,4)]
            else:
                self.test_dataset = CreatePytorchDataset(X_np,df.filename)

    def check_format(self, df):
        if df.isnull().any().any():
            raise ValueError("There are NaN values in the data.")
        if 'npimage' in df.columns and df.npimage.iloc[0].shape != (self.L, self.L, 3):
            raise ValueError("Images have the incorrect shape")

    def test_dataloader(self):

        if self.TTA:
            loader = CombinedLoader({rot : DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,drop_last=False) for rot,ds in zip(["0","90","180","270"],self.test_dataset)})
        else:
            loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,drop_last=False)
        return loader

if __name__ == '__main__':
    dm = ZooplanktonDataModule('/path/to/data', TTA=True)
    dm.setup('test')
    test_loader = dm.test_dataloader()