###########
# IMPORTS #
###########
import os
import pathlib
import sys
import lightning as pl
from time import time
from .models.model import Plankformer
from .data.datamodule import ZooplanktonDataModule
from .helpers.argparser import argparser

import torch
import logging
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
# Start timing the script
time_begin = time()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

###############
# MAIN SCRIPT #
###############

if __name__ == '__main__':
    print('\nRunning', sys.argv[0], sys.argv[1:])

    # Parse Arguments for training
    parser = argparser()
    args = parser.parse_args()
    default_scratch = os.getenv('SCRATCH')
    args.train_outpath = os.path.join(default_scratch, args.train_outpath)
    # Create Output Directory if it doesn't exist
    pathlib.Path(args.train_outpath).mkdir(parents=True, exist_ok=True)

    # Initialize the Data Module
    datamodule = ZooplanktonDataModule(
        datapath=args.datapath,
        L=args.L,
        resize_images=args.resize_images,
        TTA=args.TTA,
        batch_size=args.batch_size,
        dataset=args.dataset
    )
    datamodule.setup("fit")

    # Initialize the Model
    model = Plankformer(**vars(args))
    model.load_datamodule(datamodule)

    # Move the model to GPU if available and specified
    if torch.cuda.is_available() and args.use_gpu:
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)

    # Initialize the loggers

    callbacks=[]
    if args.use_wandb:
        import wandb
        wandb.login(key=os.environ["WANDB_API_KEY"])
        logger = WandbLogger(
            project=args.dataset,
            log_model=False
        )
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    else:
        logger = CSVLogger(save_dir=args.train_outpath, name='csv_logs')

    callbacks.append(ModelCheckpoint(
        dirpath=args.train_outpath,
        filename='best_model',
        monitor='val_loss',
        mode='min'
    ))
    # Initialize the Trainer
    trainer = pl.Trainer(logger=logger, max_epochs=args.max_epochs, callbacks=callbacks, check_val_every_n_epoch=10)

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Calculate and log the total time taken for training
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for training (in secs): {}'.format(total_secs))
