
import numpy as np
import timm


def setup_model( finetune, num_classes, **kwargs):
    """
    Set up and return the specified model architecture.

    Args:
        architecture (str): The model architecture to use.
        main_param_path (str): Path to the directory containing main parameters.
        ensemble (bool): Whether to use model ensembling.
        finetune (bool): Whether to finetune the model or use it as is.
        dataset (str): The name of the dataset.
        testing (bool, optional): Set to True if in testing mode. Defaults to False.
        train_first (bool, optional): Set to True to train the first layer of the model. Defaults to False.

    Returns:
        model: The configured model.
    """


    model = timm.create_model('beitv2_base_patch16_224.in1k_ft_in22k_in1k', pretrained=finetune, num_classes=num_classes)

    set_trainable_params(model, finetune=finetune)

    # Total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    return model

def set_trainable_params(model, train_first=False, finetune=True):
    """
    Set the trainable parameters of the model.

    Args:
        model (nn.Module): The model to configure.
        train_first (bool, optional): If True, train the first layer of the model. Defaults to False.
        finetune (bool, optional): If True, finetune the model. Defaults to True.
    """

    n_layer = 0

    for param in model.parameters():
        n_layer += 1
        param.requires_grad = False

    for i, param in enumerate(model.parameters()):
        if i < 1:
            param.requires_grad = True
        if i + 1 > n_layer - 2:
            param.requires_grad = True
        if not finetune:
            param.requires_grad = True



