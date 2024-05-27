import argparse

def argparser():
    parser = argparse.ArgumentParser(description='Configure and run the machine learning model for image classification.')

    # Paths and directories
    parser.add_argument('--datapath', nargs='*', default='training', help='Folder containing the tar training data')
    parser.add_argument('--train_outpath', default='train_out', help='Output path for training artifacts')
    parser.add_argument('--test_path', nargs='*', default=['images'], help='Data directory for testing')
    parser.add_argument('--main_param_path', default='./params/', help='Main directory where the training parameters are saved')
    parser.add_argument('--test_outpath', default='./preds/', help='Directory where you want to save the predictions')
    parser.add_argument('--dataset', default='zoo', help='Name of the dataset')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')

    # Model configuration and training options
    parser.add_argument('--random_rot', action='store_false', help='Enable random rotation during training')
    parser.add_argument('--AugMix', action='store_false', help='Enable AugMix augmentation')
    parser.add_argument('--use_data_moments', action='store_true', help='Use data moments for normalization')
    parser.add_argument('--use_scheduler', action='store_false', help='Use learning rate scheduler')
    parser.add_argument('--balance_weight', action='store_true', help='Balance class weights in loss computation')
    parser.add_argument('--priority_classes', action='store_true', help='Use priority classes for training')
    parser.add_argument('--rest_classes', action='store_true', help='Use rest classes for training')
    # Deep learning model specifics
    parser.add_argument('--architecture', default='beitv2', help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma value for focal loss')
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--swa', action='store_true', help='Use Stochastic Weight Averaging')
    parser.add_argument('--loss', choices=["cross_entropy","focal"], default="cross_entropy", help='Loss function for training')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training')
    parser.add_argument('--lr_factor', type=float, default=0.01, help='Learning rate factor for training of full body')
    parser.add_argument('--train_first', action='store_false', help='Train the first layer of the model too')
    parser.add_argument('--no_gpu', action='store_true', help='Use no GPU for training, default is False')


    # Augmentation and training/testing specifics
    parser.add_argument('--ensemble', action='store_true', help='Enable model ensembling')
    parser.add_argument('--TTA', action='store_true', help='Enable test-time augmentation')
    parser.add_argument('--testing', action='store_true', help='Set this to True if in testing mode, False for training')

    return parser

# Example of using the argument parser
if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    print(args)
