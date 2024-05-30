import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary, StochasticWeightAveraging
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import json



def output_results(outpath, im_names, labels, scores):
    """
    Output the prediction results to a file.

    Args:
        outpath (str): Output directory path.
        im_names (list): List of image filenames.
        labels (list): List of predicted labels.
    """

    labels = labels.tolist()
    base_filename = f"{outpath}/predictions"
    file_path = f"{base_filename}.txt"
    lines = [f"\n{img}------------------ {label}/{score}" for img, label,score in zip(im_names, labels,scores)]
    with open(file_path, "w") as f:
        f.writelines(lines)


def gmean(input_x, dim):
    """
    Compute the geometric mean of the input tensor along the specified dimension.

    Args:
        input_x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the geometric mean.

    Returns:
        torch.Tensor: Geometric mean of the input tensor.
    """
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))


def plot_confusion_matrix(all_labels, all_preds, class_names):
    """
    Plot and return confusion matrices (absolute and normalized).

    Args:
        all_labels (torch.Tensor): True labels.
        all_preds (torch.Tensor): Predicted labels.
        class_names (list): List of class names.

    Returns:
        tuple: (figure for absolute confusion matrix, figure for normalized confusion matrix)
    """


    class_indices = np.arange(len(class_names))
    confusion_matrix = sklearn.metrics.confusion_matrix(all_labels.cpu(), all_preds.cpu(), labels=class_indices)
    confusion_matrix_norm = sklearn.metrics.confusion_matrix(all_labels.cpu(), all_preds.cpu(), normalize="pred", labels=class_indices)
    num_classes = confusion_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(15, 15))
    fig2, ax2 = plt.subplots(figsize=(15, 15))
    if len(class_names) != num_classes:
        print(f"Warning: Number of class names ({len(class_names)}) does not match the number of classes ({num_classes}) in confusion matrix.")
        class_names = class_names[:num_classes]
    cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=class_names)
    cm_display_norm = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix_norm, display_labels=class_names)
    cmap = cvd_colormap()
    cm_display.plot(cmap=cmap, ax=ax, xticks_rotation=90)
    cm_display_norm.plot(cmap=cmap, ax=ax2, xticks_rotation=90)
    fig.tight_layout()
    fig2.tight_layout()
    return fig, fig2

def cvd_colormap():
    """
    A color map accessible for people with color vision deficiency (CVD).
    """
    stops = [0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000]
    red = [0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764]
    green = [0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832]
    blue = [0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539]

    # Create a dictionary with color information
    cdict = {
        'red': [(stops[i], red[i], red[i]) for i in range(len(stops))],
        'green': [(stops[i], green[i], green[i]) for i in range(len(stops))],
        'blue': [(stops[i], blue[i], blue[i]) for i in range(len(stops))]
    }

    # Create the colormap
    return LinearSegmentedColormap('CustomMap', segmentdata=cdict, N=255)

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with cosine annealing and warmup.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        warmup (int): Number of warmup steps.
        max_iters (int): Total number of iterations.

    Methods:
        get_lr: Compute the learning rate at the current step.
        get_lr_factor: Compute the learning rate factor at the current step.
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch >= self.max_num_iters:
            lr_factor *= self.max_num_iters / epoch
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def define_priority_classes(priority_classes):
    class_map = {class_name: i + 1 for i, class_name in enumerate(priority_classes)}
    class_map["rest"] = 0
    return class_map


def plot_score_distributions(all_scores, all_preds, class_names, true_label):
    """
    Plot the distribution of prediction scores for each class in separate plots.

    Args:
        all_scores (torch.Tensor): Confidence scores of the predictions.
        all_preds (torch.Tensor): Predicted class indices.
        class_names (list): List of class names.

    Returns:
        list: A list of figures, each representing the score distribution for a class.
    """
    # Convert scores and predictions to CPU if not already
    all_scores = all_scores.cpu().numpy()
    all_preds = all_preds.cpu().numpy()
    true_label = true_label.cpu().numpy()
    # List to hold the figures
    fig, ax = plt.subplots(len(class_names) // 4 + 1, 4, figsize=(20, len(class_names) // 4 * 5 + 1))
    ax = ax.flatten()

    # Creating a histogram for each class
    for i, class_name in enumerate(class_names):
        # Filter scores for predictions matching the current class
        sig_scores = all_scores[(true_label == i)][:, i]
        bkg_scores = all_scores[(true_label != i)][:, i]
        # Create a figure for the current class
        ax[i].hist(bkg_scores, bins=np.linspace(0, 1, 30), color="skyblue", edgecolor="black")
        ax[i].set_ylabel("Rest Counts", color="skyblue")
        ax[i].set_yscale("log")
        y_axis = ax[i].twinx()
        y_axis.hist(sig_scores, bins=np.linspace(0, 1, 30), color="crimson", histtype="step", edgecolor="crimson")
        ax[i].set_title(f"{class_name}")
        ax[i].set_xlabel("Predicted Probability")
        y_axis.set_ylabel("Signal Counts", color="crimson")
        y_axis.set_yscale("log")
    fig.tight_layout()
    return fig


def TTA_collate_fn(batch: dict):
    """
    Collate function for test time augmentation (TTA).

    Args:
        batch (dict): Dict of tuples containing images and labels.

    Returns:
        batch_images: All rotations stacked row-wise
        batch_labels: Labels of the images
    """
    batch_images = {rot: [] for rot in ["0", "90", "180", "270"]}
    batch_labels = []
    for rotated_images, label in batch:
        for rot in batch_images:
            batch_images[rot].append(rotated_images[rot])
        batch_labels.append(label)
    batch_images = {rot: torch.stack(batch_images[rot]) for rot in batch_images}
    batch_labels = torch.tensor(batch_labels)
    return batch_images, batch_labels



def plot_loss_acc(logger):
    """
    Plots the training and validation loss and accuracy from the logger's metrics file.

    Args:
        logger (Logger): The logger object containing the save directory, name, and version.

    Saves:
        loss_accuracy.png: A plot of the training and validation loss and accuracy over steps.
    """
    # Read the CSV file
    metrics_file = f"{logger.save_dir}/{logger.name}/version_{logger.version}/metrics.csv"
    metrics = pd.read_csv(metrics_file)

    # Plot the training loss
    step = metrics["step"]
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    train_acc = metrics["train_acc"]
    val_acc = metrics["val_acc"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(step[train_loss == train_loss], train_loss[train_loss == train_loss], label="Training Loss", color="skyblue")
    ax[0].plot(step[val_loss == val_loss], val_loss[val_loss == val_loss], label="Validation Loss", color="crimson")
    ax[0].set_xlabel("Step")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss vs Steps")
    ax[0].legend()

    ax[1].plot(step[train_loss == train_loss], train_acc[train_loss == train_loss], label="Training Accuracy", color="skyblue")
    ax[1].plot(step[val_loss == val_loss], val_acc[val_loss == val_loss], label="Validation Accuracy", color="crimson")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy vs Steps")
    ax[1].legend()
    fig.tight_layout()
    plt.savefig(f"{logger.save_dir}/{logger.name}/version_{logger.version}/loss_accuracy.png")

def setup_callbacks(priority_classes, ckpt_name):
    """
    Sets up callbacks for the training process.

    Args:
        priority_classes (list): List of priority classes to monitor for false positives.
        ckpt_name (str): The name of the checkpoint file.

    Returns:
        list: A list of configured callbacks including EarlyStopping, ModelCheckpoint, and ModelSummary.
    """
    callbacks = []
    monitor = "val_acc" if priority_classes == "" else "val_false_positives"
    mode = "max" if not priority_classes == "" else "min"
    callbacks.append(EarlyStopping(monitor=monitor, patience=3, mode=mode))
    callbacks.append(ModelCheckpoint(filename=ckpt_name, monitor=monitor, mode=mode))
    callbacks.append(ModelSummary())
    return callbacks

def plot_reduced_classes(model, priority_classes):
    """
    Plots the confusion matrix for reduced classes.

    Args:
        model (LightningModule): The trained model.
        priority_classes (list): List of priority classes.

    Saves:
        reduced_confusion_matrix.png: A confusion matrix of the reduced classes.
        reduced_confusion_matrix_norm.png: A normalized confusion matrix of the reduced classes.
    """
    reduced_class_map = {v: k + 1 for k, v in enumerate(priority_classes)}
    reduced_class_map["rest"] = 0
    inv_reduced_class_map = {v: k for k, v in reduced_class_map.items()}
    reduced_preds = []
    reduced_labels = []
    preds = torch.cat(model.test_step_predictions)
    true_labels = torch.cat(model.test_step_targets)
    for pred, true in zip(preds, true_labels):
        name = model.inverted_class_map[pred.item()]
        name2 = model.inverted_class_map[true.item()]
        reduced_preds.append(reduced_class_map[name] if name in reduced_class_map else 0)
        reduced_labels.append(reduced_class_map[name2] if name2 in reduced_class_map else 0)
    all_preds = torch.tensor(reduced_preds)
    all_labels = torch.tensor(reduced_labels)
    fig, fig2 = plot_confusion_matrix(all_labels, all_preds, inv_reduced_class_map)
    fig.savefig(f"{model.outpath}/reduced_confusion_matrix.png")
    fig2.savefig(f"{model.outpath}/reduced_confusion_matrix_norm.png")
