import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from ..data.datamodule import PlanktonDataModule

# Define a transform without normalization
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the dataset


# Function to calculate mean and standard deviation
def calculate_mean_std(dataset,cyanos_only=False):
    """
    Calculate the mean and standard deviation of the dataset.

    Args:
        dataloader (DataLoader): DataLoader for the dataset.

    Returns:
        tuple: (mean, std) where mean and std are tensors representing the mean and standard deviation of the dataset.
    """
    if dataset=="zoo":
        cyanos_only=False
    dm = PlanktonDataModule('/beegfs/desy/user/kaechben/eawag/training/', dataset=dataset, use_data_moments=False, calc_normalisation=True,cyanos_only=cyanos_only)
    dm.setup('fit')
    dataloader = dm.train_dataloader()
    mean = 0.0
    std = 0.0
    total_images_count = 0
    labels=[]
    for images, label in dataloader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples
        labels.append(label)

    # Print and save class balance information

    print("Balances:", torch.bincount(torch.cat(labels)),torch.cat(labels).unique())
    weights = 1 / torch.bincount(torch.cat(labels)).float().sqrt()
    print("weights:", weights)
    postfix="" if not cyanos_only else "_cyanos"
    torch.save(weights, f"./params/{dataset}/class_weights{postfix}.pt")
    print("Total images:", total_images_count)

    mean /= total_images_count
    std /= total_images_count
    return mean, std

mean, std = calculate_mean_std(dataset="phyto",cyanos_only=True)
print(f"Mean: {mean}, Std: {std}")