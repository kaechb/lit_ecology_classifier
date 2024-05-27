import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def output_results(outpath,  im_names, labels):
    """
    Output the prediction results to a file.

    Args:
        outpath (str): Output directory path.
        im_names (list): List of image filenames.
        labels (list): List of predicted labels.
    """
    name2 = 'geo_mean_'
    labels = labels.tolist()
    base_filename = f'{outpath}/Ensemble_models_Plankformer_predictions_{name2}'
    file_path = f'{base_filename}.txt'
    lines = [f'\n{img}------------------ {label}\n' for img, label in zip(im_names, labels)]
    with open(file_path, 'w') as f:
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
    confusion_matrix = sklearn.metrics.confusion_matrix(all_labels.cpu(), all_preds.cpu(),labels=class_indices )
    confusion_matrix_norm = sklearn.metrics.confusion_matrix(all_labels.cpu(), all_preds.cpu(), normalize="true",labels=class_indices)
    num_classes = confusion_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(15, 15))
    fig2, ax2 = plt.subplots(figsize=(15, 15))
    if len(class_names) != num_classes:
        print(f"Warning: Number of class names ({len(class_names)}) does not match the number of classes ({num_classes}) in confusion matrix.")
        class_names = class_names[:num_classes]
    cm_display = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=class_names)
    cm_display_norm = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix_norm, display_labels=class_names)
    cm_display.plot(cmap='viridis', ax=ax, xticks_rotation=90)
    cm_display_norm.plot(cmap='viridis', ax=ax2, xticks_rotation=90)
    fig.tight_layout()
    fig2.tight_layout()
    return fig, fig2

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

def define_rest_classes(rest_classes):
    class_map = {class_name: i + 1 for i, class_name in enumerate(rest_classes)}
    class_map["rest"] = 0
    return class_map




class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def plot_score_distributions(all_scores, all_preds, class_names,true_label):
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
    true_label=true_label.cpu().numpy()
    # List to hold the figures
    figures = []
    fig,ax=plt.subplots(3,4,figsize=(20,15))
    ax=ax.flatten()
    # Creating a histogram for each class
    for i, class_name in enumerate(class_names):

        # Filter scores for predictions matching the current class
        sig_scores = all_scores[ (true_label == i)][:,i]
        bkg_scores = all_scores[(true_label != i)][:,i]
        # Create a figure for the current class
        ax[i].hist(bkg_scores, bins=np.linspace(0,1,30), color='skyblue', edgecolor='black')
        ax[i].set_ylabel('Rest Density', color='skyblue')
        ax[i].set_yscale('log')
        y_axis = ax[i].twinx()
        y_axis.hist(sig_scores, bins=np.linspace(0,1,30), color='red',histtype='step',edgecolor='red')
        ax[i].set_title(f'{class_name}')
        ax[i].set_xlabel('Confidence Score')
        y_axis.set_ylabel('Signal Density', color='red')
        y_axis.set_yscale('log')
    fig.tight_layout()
    return fig