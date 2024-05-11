
import timm
import torch
import torch.nn as nn
import numpy as np
def setup_model( architecture, add_layer, main_param_path, dropout_1,dropout_2, fc_node,
                 last_layer_finetune,testing=False, **kwargs):
    classes = np.load(main_param_path + '/classes.npy')
    num_classes=len(np.unique(classes))

    if architecture == 'deit':
        model = timm.create_model('deit_base_distilled_patch16_224.fb_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'efficientnetb2':
        model = timm.create_model('tf_efficientnet_b2.ns_jft_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'efficientnetb5':
        model = timm.create_model('tf_efficientnet_b5.ns_jft_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'efficientnetb6':
        model = timm.create_model('tf_efficientnet_b6.ns_jft_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'efficientnetb7':
        model = timm.create_model('tf_efficientnet_b7.ns_jft_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'densenet':
        model = timm.create_model('densenet161.tv_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'mobilenet':
        model = timm.create_model('mobilenetv3_large_100.miil_in21k_ft_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'inception':
        model = timm.create_model('inception_v4.tf_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'vit':
        model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'mae':
        model = timm.create_model('vit_base_patch16_224.mae', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'swin':
        model = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    elif architecture == 'beit':
        model = timm.create_model('beit_base_patch16_224.in22k_ft_in22k_in1k', pretrained=not testing,
                                        num_classes=num_classes)
    else:
        print('This model cannot be imported. Please check from the list of models')

    # additional layers
    if add_layer:
        in_features = model.get_classifier()[-1].in_features if architecture == 'deit' else model.get_classifier().in_features

        pretrained_layers = list(model.children())[:-2] if architecture == 'deit' else list(model.children())[:-1]
        additional_layers = nn.Sequential(
                                    nn.Dropout(p=dropout_1),
                                    nn.Linear(in_features=in_features, out_features=fc_node),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=dropout_2),
                                    nn.Linear(in_features=fc_node, out_features=num_classes),
                                    )
        model = nn.Sequential(*pretrained_layers, additional_layers)
    set_trainable_params(model,add_layer, last_layer_finetune)
    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")


    return model

def set_trainable_params(model, add_layer, last_layer_finetune):
    """
      This function sets the trainable parameters of the model,
      if its last_layer_finetune is set to True, then only the last layer is trainable
    """
    n_layer = 0
    if last_layer_finetune:
        for param in model.parameters():
            n_layer += 1
            param.requires_grad = False

    for i, param in enumerate(model.parameters()):

        if i + 1 > n_layer - 2:
            param.requires_grad = True
        elif (i + 1 > n_layer - 5) and add_layer:
            param.requires_grad = True
