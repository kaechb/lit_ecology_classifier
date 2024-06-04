import argparse
import os

def argparser():
    """
    Creates an argument parser for configuring, training, and running the machine learning model for image classification.

    Arguments:
    --datapath: str
        Path to the tar file containing the training data. Default is "/store/empa/em09/aquascope/phyto.tar".
    --train_outpath: str
        Output path for training artifacts. Default is "./train_out".
    --main_param_path: str
        Main directory where the training parameters are saved. Default is "./params/".
    --dataset: str
        Name of the dataset. Default is "phyto".
    --use_wandb: flag
        Use Weights and Biases for logging. Default is False.

    --priority_classes: str
        Path to the JSON file specifying priority classes for training. Default is an empty string.
    --rest_classes: str
        Path to the JSON file specifying rest classes for training. Default is an empty string.
    --balance_classes: flag
        Balance the classes for training. Default is False.
    --batch_size: int
        Batch size for training. Default is 64.
    --max_epochs: int
        Number of epochs to train. Default is 20.
    --lr: float
        Learning rate for training. Default is 1e-2.
    --lr_factor: float
        Learning rate factor for training of full body. Default is 0.01.
    --no_gpu: flag
        Use no GPU for training. Default is False.
    --testing: flag
        Set this to True if in testing mode, False for training. Default is False.

    Returns:
        argparse.ArgumentParser: The argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser(description="Configure, train and run the machine learning model for image classification.")

    # Paths and directories
    parser.add_argument("--datapath",  default="/store/empa/em09/aquascope/phyto.tar", help="Path to the tar file containing the training data")
    parser.add_argument("--train_outpath", default="./train_out", help="Output path for training artifacts")
    parser.add_argument("--main_param_path", default="./params/", help="Main directory where the training parameters are saved")
    parser.add_argument("--dataset", default="phyto", help="Name of the dataset")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights and Biases for logging")

    # Model configuration and training options
    parser.add_argument("--priority_classes", type=str, default="", help="Path to the JSON file specifying priority classes for training")
    parser.add_argument("--rest_classes", type=str, default="", help="Path to the JSON file specifying rest classes for training")
    parser.add_argument("--balance_classes", action="store_true", help="Balance the classes for training")
    parser.add_argument("--batch_size", type=int, default=180, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for training")
    parser.add_argument("--lr_factor", type=float, default=0.01, help="Learning rate factor for training of full body")
    parser.add_argument("--no_gpu", action="store_true", help="Use no GPU for training, default is False")
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy", help="Loss function to use")

    # Augmentation and training/testing specifics
    parser.add_argument("--testing", action="store_true", help="Set this to True if in testing mode, False for training")
    parser.add_argument("--no_TTA", action="store_true", help="Enable Test Time Augmentation")
    return parser

def inference_argparser():
    """
    Creates an argument parser for using the classifier on unlabeled data.

    Arguments:
    --batch_size: int
        Batch size for inference. Default is 180.
    --outpath: str
        Directory where predictions will be saved. Default is "./preds/".
    --model_path: str
        Path to the model checkpoint file. Default is "./checkpoints/model.ckpt".
    --datapath: str
        Path to the tar file containing the data to classify. Default is "/store/empa/em09/aquascope/phyto.tar".
    --no_gpu: flag
        Use no GPU for inference. Default is False.
    --no_TTA: flag
        Disable test-time augmentation. Default is False.
    --gpu_id: int
        GPU ID to use for inference. Default is 0.
    --limit_pred_batches: int
        Limit the number of batches to predict. Default is 0, meaning no limit, set a low number to debug.
    --prog_bar: flag
        Enable progress bar. Default is False.
    Returns:
        argparse.ArgumentParser: The argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser(description="Use Classifier on unlabeled data.")
    parser.add_argument("--batch_size", type=int, default=180, help="Batch size for inference")
    parser.add_argument("--outpath", default="./preds/", help="Directory where predictions will be saved")
    parser.add_argument("--model_path", default="./checkpoints/model.ckpt", help="Path to the model checkpoint file")
    parser.add_argument("--datapath",  default="/store/empa/em09/aquascope/phyto.tar", help="Path to the tar file containing the data to classify")
    parser.add_argument("--no_gpu", action="store_true", help="Use no GPU for inference, default is False")
    parser.add_argument("--no_TTA", action="store_true", help="Disable test-time augmentation")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for inference")
    parser.add_argument("--prog_bar", action="store_true", help="Enable progress bar")
    parser.add_argument("--limit_pred_batches", type=int, default=0, help="Limit the number of batches to predict")
    return parser


# Example of using the argument parser
if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    print(args)
