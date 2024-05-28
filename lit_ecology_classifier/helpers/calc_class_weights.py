import logging
import pprint

import torch



def calculate_class_weights(datamodule):
    """

    Calculate and save class weights and the mean and standard deviation of the dataset.

    Args:
        dataloader (DataLoader): DataLoader for the dataset.

    Returns:
        tuple: (mean, std) where mean and std are tensors representing the mean and standard deviation of the dataset.
    """

    logging.info("Calculating class weights...")
    dataloader = datamodule.train_dataloader()
    mean = 0.0
    std = 0.0
    total_images_count = 0
    labels=[]
    for images, label in dataloader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
        labels.append(label)

    # Print and save class balance information
    logging.info("Balances:", pprint.pformat(torch.bincount(torch.cat(labels)),torch.cat(labels).unique()))
    # Empirical studies from Jean-Oliver Irisson suggest that the square root of the class weights is a good starting point
    weights = 1 / torch.bincount(torch.cat(labels)).float().sqrt()
    logging.info("weights:", pprint.pformat(weights))
    return weights


