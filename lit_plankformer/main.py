###########
# IMPORTS #
###########
import os
import pathlib
import sys
import lightning as pl
from time import time
from .models.model import Plankformer
from .data.datamodule import PlanktonDataModule
from .helpers.argparser import argparser
import json
import torch
import logging
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, ModelSummary, EarlyStopping
from lightning.pytorch.tuner import Tuner

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
    default_scratch = "/beegfs/desy/user/kaechben/eawag/"
    args.train_outpath = os.path.join(default_scratch, args.train_outpath)
    args.datapath = os.path.join(default_scratch, args.datapath)
    print(args.datapath)

    # Create Output Directory if it doesn't exist
    pathlib.Path(args.train_outpath).mkdir(parents=True, exist_ok=True)
    gpus = torch.cuda.device_count() if not args.no_gpu else 0

    if args.priority_classes:
        args.priority_classes=json.load(open("./lit_plankformer/data/" + 'priority.json'))["priority_classes"]
    else:
        args.priority_classes=[]
    if args.rest_classes:
        args.rest_classes=json.load(open("./lit_plankformer/data/" + 'rest.json'))["priority_classes"]
    else:
        args.rest_classes=[]
    print("Priority classes:",args.priority_classes)
    # Initialize the Data Module
    datamodule = PlanktonDataModule(**vars(args))
    datamodule.setup("fit")

    # Initialize the loggers
    callbacks = []

    if args.use_wandb:
        logger = WandbLogger(
            project=args.dataset,
            log_model=False,
            save_dir=args.train_outpath,
        )
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        logger.experiment.log_code("./lit_plankformer",include_fn=lambda path: path.endswith(".py"))
    else:
        logger = CSVLogger(save_dir=args.train_outpath, name='csv_logs')

    torch.backends.cudnn.allow_tf32 = False

    # Initialize the Model
    # args.lr=args.lr * args.lr_factor
    model = Plankformer(**vars(args), finetune=True)
    model.load_datamodule(datamodule)

    # Move the model to GPU if available and specified
    callbacks.append(ModelCheckpoint(filename='best_model_acc_stage1', monitor='val_acc', mode='max'))

    # Initialize the Trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=10,
        devices=gpus,
        strategy='ddp' if gpus > 1 else "auto",
        enable_progress_bar=False,
        default_root_dir=args.train_outpath
    )
    # Train the model
    trainer.fit(model, datamodule=datamodule)
    model = Plankformer.load_from_checkpoint(
        str(trainer.checkpoint_callback.best_model_path),
        lr=args.lr * args.lr_factor,
        finetune=False
    )
    model.load_datamodule(datamodule)
    callbacks = [LearningRateMonitor(logging_interval='step')] if args.use_wandb else []
    callbacks.append(EarlyStopping(monitor='val_acc', patience=100, mode='max'))
    if args.swa:
        callbacks.extend([
            StochasticWeightAveraging(swa_lrs=1e-3),
            ModelCheckpoint(filename='best_model_epoch_stage2', monitor='epoch', mode='max')
        ])
    callbacks.append(ModelCheckpoint(filename='best_model_acc_stage2', monitor='val_acc' if not len(datamodule.priority_classes)==0 else 'val_false_positives', mode='max' if not len(datamodule.priority_classes)==0 else 'min'))
    callbacks.append(ModelSummary())

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=2 * args.max_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=5,
        devices=gpus,
        strategy='ddp' if gpus > 1 else "auto",
        enable_progress_bar=False,
        default_root_dir=args.train_outpath
    )

    trainer.fit(model, datamodule=datamodule)

    # Calculate and log the total time taken for training
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for training (in secs): {}'.format(total_secs))
