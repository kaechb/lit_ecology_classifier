###########
# IMPORTS #
###########

import pathlib
import sys
import lightning as pl
from time import time
import numpy as np
from models.model import Plankformer
from data.datamodule import PlanktonDataModule
from helpers.argparser import argparser

import torch
import logging

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
    parser = argparser()
    args = parser.parse_args()

    # Create Output Directory if it doesn't exist
    pathlib.Path(args.test_outpath).mkdir(parents=True, exist_ok=True)

    # Initialize the Data Module
    data_module = PlanktonDataModule(
        datapath=args.test_path,
        L=args.L,
        resize_images=args.resize_images,
        TTA=args.TTA,
        batch_size=args.batch_size,
        dataset=args.dataset
    )
    data_module.setup("test")

    # Initialize the Model
    model = Plankformer(**vars(args))
    model.load_datamodule(data_module)

    # Move the model to GPU if available and specified
    if torch.cuda.is_available() and args.use_gpu:
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)

    # Initialize the Trainer and Perform Predictions
    trainer = pl.Trainer()
    trainer.test(model, datamodule=data_module)

    # Calculate and log the total time taken for prediction
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for prediction (in secs): {}'.format(total_secs))
