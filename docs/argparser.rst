Argument Parser Documentation
=============================

This document details the command-line arguments for configuring, training, running, and inferring with the machine learning model for image classification.

Main Argument Parser
--------------------

The main argument parser is used to configure, train, and run the model.

.. code-block:: python

    def argparser():
        parser = argparse.ArgumentParser(description="Configure, train and run the machine learning model for image classification.")

        # Paths and directories
        parser.add_argument("--datapath",  default="/store/empa/em09/aquascope/phyto.tar", help="Folder containing the tar training data")
        parser.add_argument("--train_outpath", default="./train_out", help="Output path for training artifacts")
        parser.add_argument("--main_param_path", default="./params/", help="Main directory where the training parameters are saved")
        parser.add_argument("--dataset", default="zoo", help="Name of the dataset")
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

### Arguments

- `--datapath` (default: `/store/empa/em09/aquascope/phyto.tar`): Folder containing the tar training data.
- `--train_outpath` (default: `./train_out`): Output path for training artifacts.
- `--main_param_path` (default: `./params/`): Main directory where the training parameters are saved.
- `--dataset` (default: `zoo`): Name of the dataset.
- `--use_wandb` (default: `False`): Use Weights and Biases for logging.
- `--priority_classes` (default: `""`): Use priority classes for training, specify the path to the JSON file.
- `--balance_classes` (default: `False`): Balance the classes for training.
- `--batch_size` (default: `64`): Batch size for training.
- `--max_epochs` (default: `20`): Number of epochs to train.
- `--lr` (default: `1e-2`): Learning rate for training.
- `--lr_factor` (default: `0.01`): Learning rate factor for training of full body.
- `--no_gpu` (default: `False`): Use no GPU for training.
- `--testing` (default: `False`): Set this to True if in testing mode, False for training.

Inference Argument Parser
-------------------------

The inference argument parser is used to classify unlabelled data.

.. code-block:: python

    def inference_argparser():
        parser = argparse.ArgumentParser(description="Use Classifier on unlabelled data.")
        parser.add_argument("--outpath", default="./preds/", help="Directory where you want to save the predictions")
        parser.add_argument("--model_path", default="./checkpoints/model.ckpt", help="Path to the model file")
        parser.add_argument("--datapath",  default="", help="Path to the folder containing the data to classify as Tar file")
        parser.add_argument("--no_gpu", action="store_true", help="Use no GPU for training, default is False")
        parser.add_argument("--no_TTA", action="store_true", help="Disable test-time augmentation")

        return parser

### Arguments

- `--outpath` (default: `./preds/`): Directory where you want to save the predictions.
- `--model_path` (default: `./checkpoints/model.ckpt`): Path to the trained model file to be used for inference.
- `--datapath` (default: `""`): Path to the folder containing the data to classify as Tar file.
- `--no_gpu` (default: `False`): Use no GPU for training.
- `--no_TTA` (default: `False`): Disable test-time augmentation.
