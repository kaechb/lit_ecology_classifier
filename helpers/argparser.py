import argparse
def argparser():

    parser = argparse.ArgumentParser(description='Configure and run the machine learning model for image classification.')

    # Paths and directories
    parser.add_argument('--datapaths', nargs='*', default=['/home/EAWAG/chenchen/data/Zooplankton/train_data/training_zooplankton_new_220823///'], help='Input data paths')
    parser.add_argument('--outpath', default='/home/EAWAG/chenchen/out/train_out/Zooplankton/20230801/BEiT///', help='Output path for training artifacts')
    parser.add_argument('--test_path', nargs='*', default=['/scratch/snx3000/bkch/1622498401/images'], help="Data Directory")
    parser.add_argument('--main_param_path', default='./params/', help="Main directory where the training parameters are saved")
    parser.add_argument('--test_outpath', default='./preds/', help="Directory where you want to save the predictions")
    parser.add_argument('--model_path', nargs='*', default=['./ckpts/01', './ckpts/02', './ckpts/03'], help='Paths of the saved models')

    # Model configuration and training options
    parser.add_argument('--aug', type=bool, default=True, help='Enable training augmentation')
    parser.add_argument('--L', type=int, default=128, help='Image size ')
    parser.add_argument('--finetuned', choices=["original","tuned","finetuned"], default="tuned", help='Choose "0" or "1" or "2" for finetuning')
    parser.add_argument('--testSplit', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--class_select', default=None, help='Specific class to select for training/testing')
    parser.add_argument('--classifier', default='multi', help='Type of classifier to use')
    parser.add_argument('--balance_weight', default='yes', help='Balance weights in loss computation')
    parser.add_argument('--datakind', default='image', help='Type of data to process')
    parser.add_argument('--ttkind', default='image', help='Type of test-time augmentation to use')

    # Augmentation and training/testing specifics
    parser.add_argument('--resize_images', type=int, default=1, help='Control resizing of images')
    parser.add_argument('--compute_extrafeat', default=False, type=bool, help='Compute extra features')
    parser.add_argument('--dataset_name', default='zoolake', help='Name of the dataset')

    # Deep learning model specifics
    parser.add_argument('--architecture', default='beit', help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID for training")
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for training')
    parser.add_argument('--finetune_lr', type=float, default=1e-05, help='Learning rate for fine-tuning')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.03, help='Weight decay rate')
    parser.add_argument('--clip_grad_norm', type=float, default=0, help='Clip gradient norms')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU for training')
    parser.add_argument('--finetune', type=int, default=1, help='Enable fine-tuning')
    parser.add_argument('--finetune_epochs', type=int, default=400, help='Number of epochs for fine-tuning')
    parser.add_argument('--ensemble', type=bool, default=True, help='Enable model ensembling')
    parser.add_argument('--threshold', type=float, default=0.0, help='Threshold value for predictions')
    parser.add_argument('--TTA', type=bool, default=True, help='Enable test-time augmentation')
    parser.add_argument('--predict', default=None, help='Predict using the model')
    parser.add_argument('--last_layer_finetune', default='yes', help='Finetune the last layer of the model')
    parser.add_argument('--add_layer', type=bool, default=False, help='Add additional layers to the model')
    parser.add_argument('--dropout_1', type=float, default=0.0, help='Dropout rate for the first dropout layer')
    parser.add_argument('--dropout_2', type=float, default=0.0, help='Dropout rate for the second dropout layer')
    parser.add_argument('--fc_node', type=int, default=0, help='Number of nodes in the fully connected layer')
    parser.add_argument('--testing', type=bool, default=True, help='Set this to True if in testing mode, False for training')
    parser.add_argument('--TTA_type', type=int, default=0, help='Type of test-time augmentation to use')

    return parser