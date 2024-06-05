###########
# IMPORTS #
###########

import logging
import pathlib
import sys
from time import time
import pprint

import lightning as pl
import torch

from .data.datamodule import DataModule
from .helpers.argparser import inference_argparser
from .models.model import LitClassifier
from lit_ecology_classifier.helpers.helpers import plot_confusion_matrix, plot_reduced_classes

# Start timing the script
time_begin = time()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############
# MAIN SCRIPT #
###############

if __name__ == '__main__':
    print('\nRunning', sys.argv[0], sys.argv[1:])

    # Parse Arguments for prediction
    parser = inference_argparser()
    args = parser.parse_args()

    # Create Output Directory if it doesn't exist
    pathlib.Path(args.outpath).mkdir(parents=True, exist_ok=True)

    # Initialize the Model
    model = LitClassifier.load_from_checkpoint(args.model_path)

    # Initialize the Data Module
    hparams = model.hparams # copy the hyperparameters from the model
    model.hparams.batch_size = args.batch_size
    model.hparams.TTA = not args.no_TTA # set the TTA flag based on the argument
    model.hparams.outpath = args.outpath
    model.hparams.datapath = args.datapath
    model.hparams.use_wandb = False
    # model.hparams.priority_classes = "config/priority.json" #TODO remove this
    data_module = DataModule(**model.hparams)
    data_module.setup("predict")

    model.load_datamodule(data_module)

    # Initialize the Trainer and Perform Predictions
    trainer = pl.Trainer(devices=torch.cuda.device_count() if not args.no_gpu else 0, strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        enable_progress_bar=True, default_root_dir=args.outpath)
    trainer.test(model, datamodule=data_module)

    # Calculate and log the total time taken for prediction
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for prediction (in secs): {}'.format(total_secs))
    # model.hparams.priority_classes="config/priority.json"
    # with open(model.hparams.priority_classes, 'r') as file:
    #     priority_classes = json.load(file)["priority_classes"]
    # plot_reduced_classes(model, priority_classes)
