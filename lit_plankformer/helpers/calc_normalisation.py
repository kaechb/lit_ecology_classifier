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
dm = PlanktonDataModule('/scratch/snx3000/bkch/training/Phytolake1.tar',dataset="phyto")
dm.setup('fit')
dataloader = dm.train_dataloader()
# Function to calculate mean and standard deviation
def calculate_mean_std(dataloader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in dataloader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std

mean, std = calculate_mean_std(dataloader)
print(f"Mean: {mean}, Std: {std}")
