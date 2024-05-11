###########
# IMPORTS #
###########


import logging

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from helpers.load_images import load_images


class CreateDataset:
    def __init__(self, datapath, L=None, resize_images=None, TTA=True):
        """
            Sets up test set for evaluation of trained model
        """
        self.L = L
        self.df = load_images(datapath, L, resize_images, )
        self.check_format()  # Some sanity checks on the dataset
        self.filenames = self.df.filename
        self.X = self.df.drop(columns=['classname', 'url', 'filename', 'file_size', 'timestamp'], errors='ignore')
        self.Ximage = self.X.npimage
        self.X_np  = np.stack(self.X['npimage'].values)
        # To store the data
        self.Data = [self.filenames, [],
                         [], [], [],
                         [], [], [],
                         [], [], []]
        self.Data[1]= self.X_np
        self.filenames = [self.filenames]

        self.classes = np.load('./data/classes.npy')
        self.class_weights_tensor = torch.load('./data/class_weights_tensor.pt')
        data_train = self.Data[1].astype(np.float64)
        data_train = 255 * data_train
        self.X_train = data_train.astype(np.uint8)
        self.TTA = TTA


    def check_format(self):
        """ Basic checks on the dataset """
        # Columns potentially useful for classification
        ucols = self.df.drop(columns=['classname', 'url', 'filename', 'file_size', 'timestamp'],
                             errors='ignore').columns
        if len(ucols) < 1:
            logging.info('Columns: {}'.format(self.df.columns))
            raise ValueError('The dataset has no useful columns.')
        # Check for NaNs
        if self.df.isnull().any().any():
            logging.error('There are NaN values in the data.')
            raise ValueError("There are NaN values in the data.")
        # Check that the images have the expected size
        if 'npimage' in self.df.columns:
            if self.df.npimage[0].shape != (self.L, self.L, 3):
                logging.error(
                    'Cdata Check(): Images were not reshaped correctly: {} instead of {}'.format(self.npimage[0].shape,
                                                                                                 (self.L, self.L, 3)))
                raise ValueError("Images have the incorrect shape")


    def create_data_loaders(self):
        test_dataset = CreatePytorchDataset(X=self.X_train,)
        self.test_dataloader = DataLoader(test_dataset, 32, shuffle=False, num_workers=4,
                                          pin_memory=True)
        if self.TTA:
            test_dataset_1 = CreatePytorchDataset(X=self.X_train, rot=1)
            self.test_dataloader_1 = DataLoader(test_dataset_1, 32, shuffle=False, num_workers=4,
                                            pin_memory=True)
            test_dataset_2 = CreatePytorchDataset(X=self.X_train, rot=2)
            self.test_dataloader_2 = DataLoader(test_dataset_2, 32, shuffle=False, num_workers=4,
                                            pin_memory=True)
            test_dataset_3 = CreatePytorchDataset(X=self.X_train, rot=3)
            self.test_dataloader_3 = DataLoader(test_dataset_3, 32, shuffle=False, num_workers=4,
                                            pin_memory=True)




class CreatePytorchDataset(Dataset):
    """
    Dataset that includes a rotation by 0,90,180,270 degrees
    depending on the value of rot
    """

    def __init__(self, X , rot=0):
        """Initialization"""
        self.X = X
        self.transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(90*rot,90*rot)),
        T.ToTensor()])


    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.X[index]
        return  self.transform(image)





if __name__ == '__main__':
    pass
