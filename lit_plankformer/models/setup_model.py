
import timm
import torch
import numpy as np
import copy
def setup_model( architecture,  main_param_path,
                 ensemble,finetuned,model_path,dataset,testing=False, **kwargs):
    classes = np.load(main_param_path +"/"+dataset+ '/classes.npy')
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

    set_trainable_params(model)
    # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    if  testing:
        if ensemble:
            model = [copy.deepcopy(model) for i in range(len( model_path))]
        else:
            model = [model]
            i=0
            for m in model:
                m.load_state_dict(torch.load(model_path[i] + '/trained_model_' + finetuned + '.pth',map_location="cuda" if torch.cuda.is_available() else "cpu")['model_state_dict'])
                i+=1
    return model

def set_trainable_params(model):
    """
      This function sets the trainable parameters of the model,
      if its last_layer_finetune is set to True, then only the last layer is trainable
    """
    n_layer = 0

    for param in model.parameters():
        n_layer += 1
        param.requires_grad = False

    for i, param in enumerate(model.parameters()):

        if i + 1 > n_layer - 2:
            param.requires_grad = True
