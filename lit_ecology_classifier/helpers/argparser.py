import argparse
import os

def argparser():
    parser = argparse.ArgumentParser(description="Configure, train and run the machine learning model for image classification.")

    # Paths and directories
    parser.add_argument("--datapath",  default="/store/empa/em09/aquascope/phyto.tar", help="Folder containing the tar training data")
    parser.add_argument("--train_outpath", default="./train_out", help="Output path for training artifacts")
    parser.add_argument("--main_param_path", default="./params/", help="Main directory where the training parameters are saved")
    parser.add_argument("--dataset", default="phyto", help="Name of the dataset")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights and Biases for logging")

    # Model configuration and training options
    parser.add_argument("--priority_classes", type=str, default="", help="Use priority classes for training, specify the path to the JSON file")
    parser.add_argument("--balance_classes", action="store_true", help="Balance the classes for training")
    # Deep learning model specifics
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--max_epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for training")
    parser.add_argument("--lr_factor", type=float, default=0.01, help="Learning rate factor for training of full body")
    parser.add_argument("--no_gpu", action="store_true", help="Use no GPU for training, default is False")

    # Augmentation and training/testing specifics
    parser.add_argument("--testing", action="store_true", help="Set this to True if in testing mode, False for training")
    return parser

def inference_argparser():
    parser = argparse.ArgumentParser(description="Use Classifier on unlabelled data.")
    parser.add_argument("--outpath", default="./preds/", help="Directory where you want to save the predictions")
    parser.add_argument("--model_path", default="./checkpoints/model.ckpt", help="Path to the model file")
    parser.add_argument("--datapath",  default="/store/empa/em09/aquascope/phyto.tar", help="Path to the folder containing the data to classify as Tar file")
    parser.add_argument("--no_gpu", action="store_true", help="Use no GPU for training, default is False")
    parser.add_argument("--no_TTA", action="store_true", help="Disable test-time augmentation")

    return parser


# Example of using the argument parser
if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    print(args)
