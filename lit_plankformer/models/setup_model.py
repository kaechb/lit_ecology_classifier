
import timm
import numpy as np



def setup_model(architecture, main_param_path, ensemble, finetune, dataset, testing=False, train_first=False, **kwargs):
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
    classes = np.load(main_param_path + "/" + dataset + '/classes.npy')
    num_classes = len(np.unique(classes))

    if architecture == 'deit':
        model = timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'deit_distilled':
        model = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'efficientnetb2':
        model = timm.create_model('tf_efficientnet_b2.ns_jft_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'efficientnetb5':
        model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'efficientnetb6':
        model = timm.create_model('tf_efficientnet_b6.ns_jft_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'efficientnetb7':
        model = timm.create_model('tf_efficientnet_b7.ns_jft_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'densenet':
        model = timm.create_model('densenet121', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'mobilenet':
        model = timm.create_model('mobilenetv3_large_100.miil_in21k_ft_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'inception':
        model = timm.create_model('inception_v4.tf_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'vit':
        model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'mae':
        model = timm.create_model('vit_base_patch16_224.mae', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'swin':
        model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'beit':
        model = timm.create_model('beit_base_patch16_224.in22k_ft_in22k_in1k', pretrained=finetune, num_classes=num_classes)
    elif architecture == 'beitv2':
        model = timm.create_model('beitv2_base_patch16_224.in1k_ft_in22k_in1k', pretrained=finetune, num_classes=num_classes)
    else:
        print('This model cannot be imported. Please check from the list of models')
        return None

    set_trainable_params(model, train_first=train_first, finetune=finetune)

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
        if train_first and i < 1:
            param.requires_grad = True
        if i + 1 > n_layer - 2:
            param.requires_grad = True
        if not finetune:
            param.requires_grad = True



