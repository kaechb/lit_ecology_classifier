###########
# IMPORTS #
###########

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import pickle
import pandas as pd
import glob
import glob
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import logging
from helpers.image_edits import ResizeWithProportions, ResizeWithoutProportions
from helpers.compute_extra_features import compute_extrafeat_function

import joblib
import logging
import matplotlib.pyplot as plt







class CreateDataset:
    def __init__(self, datapath, L=None, compute_extrafeat=None, resize_images=None,
                   training_data=True):
        """
        Loads dataset using the function in the Cdata class.
        The flag `training_data` is there because of how the taxonomists created the data directories.
        In the folders that I use for training there is an extra subfolder called `training_data`. This subfolder is absent, for example, in the validation directories.
        """
        self.L = L
        self.df = LoadImages(datapath, L, resize_images, )
        self.Check()  # Some sanity checks on the dataset
        self.CreateXy()  # Creates X and y, i.e. features and labels



    def CreateXy(self):
        """
        Creates features and target
        - removing the evidently junk columns.
        - allowing to access images and features separately and confortably
        """
        self.filenames = self.df.filename
        self.X = self.df.drop(columns=['classname', 'url', 'filename', 'file_size', 'timestamp'], errors='ignore')
        self.Ximage = self.X.npimage

    def Check(self):
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

    def CreateTrainTestSets(self, train_main, test_main,  classifier=None, balance_weight=None,
                            valid_set=None, compute_extrafeat=None, random_state=12345):
        """
        Creates train and test sets using the CtrainTestSet class
        """
        # If none of the following arguments is passed, take the one from the training arguments
        classifier = classifier or train_main.classifier
        balance_weight = balance_weight or train_main.balance_weight
        compute_extrafeat = compute_extrafeat or train_main.compute_extrafeat
        # Set default value for testSplit
        self.X_np  = np.stack(self.X['npimage'].values)
        # To store the data
        self.Data = [self.filenames, [],
                         [], [], [],
                         [], [], [],
                         [], [], []]
        self.Data[1]= self.X_np
        self.Filenames = [self.filenames]



class CreateDataForPlankton:
    def __init__(self):
        pass

    def make_test_set(self, test_main, prep_data):
        Data = prep_data.Data
        self.classes = np.load(test_main.main_param_path + '/classes.npy')
        self.Filenames = prep_data.Filenames
        self.class_weights_tensor = torch.load(test_main.main_param_path + '/class_weights_tensor.pt')
        trX = Data[1]
        data_train = trX.astype(np.float64)
        data_train = 255 * data_train
        self.X_train = data_train.astype(np.uint8)



    def create_data_loaders(self, test_main):
        test_dataset = CreatePytorchDataset(X=self.X_train, TTA_type=0)
        self.test_dataloader = DataLoader(test_dataset, 32, shuffle=False, num_workers=4,
                                          pin_memory=True)
        if test_main.TTA == 'yes':
            test_dataset_1 = CreatePytorchDataset(X=self.X_train, TTA_type=1)
            self.test_dataloader_1 = DataLoader(test_dataset_1, 32, shuffle=False, num_workers=4,
                                            pin_memory=True)
            test_dataset_2 = CreatePytorchDataset(X=self.X_train, TTA_type=2)
            self.test_dataloader_2 = DataLoader(test_dataset_2, 32, shuffle=False, num_workers=4,
                                            pin_memory=True)
            test_dataset_3 = CreatePytorchDataset(X=self.X_train, TTA_type=3)
            self.test_dataloader_3 = DataLoader(test_dataset_3, 32, shuffle=False, num_workers=4,
                                            pin_memory=True)




class CreatePytorchDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, X, TTA_type):
        """Initialization"""
        self.X = X
        self.TTA_type = TTA_type

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.X)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image = self.X[index]
        TTA_type = self.TTA_type
        if TTA_type == 0:
            X = self.transform(image)
        elif TTA_type == 1:
            X = self.transform_TTA_1(image)
        elif TTA_type == 2:
            X = self.transform_TTA_2(image)
        elif TTA_type == 3:
            X = self.transform_TTA_3(image)
        sample = X
        return sample

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor()])
    transform_y = T.Compose([T.ToTensor()])

    transform_TTA_1 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(90, 90)),
        T.ToTensor()])
    transform_TTA_1_y = T.Compose([T.ToTensor()])

    transform_TTA_2 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(180, 180)),
        T.ToTensor()])
    transform_TTA_2_y = T.Compose([T.ToTensor()])

    transform_TTA_3 = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.RandomRotation(degrees=(270, 270)),
        T.ToTensor()])
    transform_TTA_3_y = T.Compose([T.ToTensor()])



def RemoveUselessCols(df):
    """ Removes columns with no information from dataframe """
    cols_to_check = df.columns.difference(['npimage'])
    # Identify columns where all values are the same, applying the check only to the specified columns
    informative_cols = df.loc[:, df[cols_to_check].nunique(dropna=False) > 1]
    # Check if 'npimage' was in the original DataFrame and add it back to the results if it was
    if 'npimage' in df.columns:
        informative_cols['npimage'] = df['npimage']
    return informative_cols


def LoadImage(filename, L=None, resize_images=None, show=False):
    """ Loads one image, and rescales it to size L.
    The pixel values are between 0 and 255, instead of between 0 and 1, so they should be normalized outside of the function
    """

    image = Image.open(filename)
    # Set image's largest dimension to target size, and fill the rest with black pixels
    if not resize_images:
        rescaled = 0
    elif resize_images == 1:
          # width and height are assumed to be the same (assertion at the beginning)
        image, rescaled = ResizeWithProportions(image,L)
    elif resize_images == 2:
         # width and height are assumed to be the same (assertion at the beginning)
        image, rescaled = ResizeWithoutProportions(image,L)
    npimage = np.array(image.copy(), dtype=np.float32)
    if show:
        plt.figure()
        os.makedirs("debug",exist_ok=True)
        plt.imshow(np.array(image))
        plt.savefig("debug/test.png")
        plt.close()
    image.close()
    return npimage, rescaled



def LoadImages(datapaths, L, resize_images=None, training_data=True):
    """
    Loads images from specified directories, processes them, and returns a DataFrame containing
    the image data and filenames.

    Arguments:
    datapaths     - List of base paths where the image data is stored.
    L             - Target size to which images are rescaled (LxL, maintaining proportions).
    resize_images - If provided, specifies additional resizing options.


    Returns:
    df            - DataFrame with columns ['filename', 'npimage'] containing image data.
    """
    # Define pattern based on directory structure.
    subfolder = 'training_data/' if training_data else ''
    file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']

    # Generate list of image paths.
    image_paths = []
    for base_path in datapaths:
        for pattern in file_patterns:
            image_paths.extend(glob.glob(os.path.join(base_path, subfolder, pattern)))

    # Load images and create DataFrame entries.
    data = []
    for image_path in image_paths:
        npimage, rescaled = LoadImage(image_path, L, resize_images,)
        data.append({'filename': image_path, 'npimage': npimage / 255.0})

    # Create and shuffle DataFrame.
    df = pd.DataFrame(data)
    return df






if __name__ == '__main__':
    pass
