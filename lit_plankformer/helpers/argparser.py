import argparse
def argparser():

    parser = argparse.ArgumentParser(description='Configure and run the machine learning model for image classification.')

    # Paths and directories
    parser.add_argument('--datapath', nargs='*', default='/scratch/snx3000/bkch/training/Phytolake1.tar', help='train data path, should be tar')
    parser.add_argument('--train_outpath', default='train_out', help='Output path for training artifacts')
    parser.add_argument('--test_path', nargs='*', default=['images'], help="Data Directory")
    parser.add_argument('--main_param_path', default='./params/', help="Main directory where the training parameters are saved")
    parser.add_argument('--test_outpath', default='./preds/', help="Directory where you want to save the predictions")
    parser.add_argument('--model_path', nargs='*', default=['./ckpts/01', './ckpts/02', './ckpts/03'], help='Paths of the saved models')
    parser.add_argument('--dataset', default='zoo', help='Name of the dataset')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')

    # Model configuration and training options
    parser.add_argument('--aug', type=bool, default=True, help='Enable training augmentation')
    parser.add_argument('--L', type=int, default=128, help='Image size ')
    parser.add_argument('--finetuned', choices=["original","tuned","finetuned"], default="tuned")
    parser.add_argument('--testSplit', type=float, default=0.2, help='Fraction of data to use for testing')

    parser.add_argument('--balance_weight', default='yes', help='Balance weights in loss computation')
    parser.add_argument('--datakind', default='image', help='Type of data to process')

    # Augmentation and training/testing specifics
    parser.add_argument('--resize_images', type=int, default=1, help='Control resizing of images')
    parser.add_argument('--compute_extrafeat', default=False, type=bool, help='Compute extra features')


    # Deep learning model specifics
    parser.add_argument('--architecture', default='beit', help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=50, help='Number of epochs to train')

    parser.add_argument('--lr', type=float, default=0.1e-5, help='Learning rate for training')
    parser.add_argument('--warmup', type=int, default=0, help='Warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.03, help='Weight decay rate')
    parser.add_argument('--clip_grad_norm', type=float, default=0, help='Clip gradient norms')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Use GPU for training')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID for training")
    parser.add_argument('--finetune', type=int, default=1, help='Enable fine-tuning')
    parser.add_argument('--finetune_epochs', type=int, default=200, help='Number of epochs for fine-tuning')
    parser.add_argument('--ensemble', type=bool, default=True, help='Enable model ensembling')
    parser.add_argument('--TTA', type=bool, default=True, help='Enable test-time augmentation')
    parser.add_argument('--predict', default=None, help='Predict using the model')
    parser.add_argument('--testing', type=bool, default=False, help='Set this to True if in testing mode, Fal