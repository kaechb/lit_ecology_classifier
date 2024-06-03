###########
# IMPORTS #
###########
import logging

from lightning.pytorch.strategies import DDPStrategy
import pathlib
import sys
from time import time

import lightning as l
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from .data.datamodule import DataModule
from .helpers.argparser import argparser
from .helpers.calc_class_weights import calculate_class_weights
from .helpers.helpers import setup_callbacks
from .models.model import LitClassifier

# Start timing the script
time_begin = time()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

###############
# MAIN SCRIPT #
###############

if __name__ == "__main__":
    print("\nRunning", sys.argv[0], sys.argv[1:])

    # Parse Arguments for training
    parser = argparser()
    args = parser.parse_args()
    # Create Output Directory if it doesn't exist
    pathlib.Path(args.train_outpath).mkdir(parents=True, exist_ok=True)
    gpus = torch.cuda.device_count() if not args.no_gpu else 0
    logging.info(f"Using {gpus} GPUs for training.")
    # Initialize the Data Module
    datamodule = DataModule(**vars(args))
    datamodule.setup("fit")

    # Initialize the loggers
    if args.use_wandb:
        logger = WandbLogger(
            project=args.dataset,
            log_model=False,
            save_dir=args.train_outpath,
        )
        logger.experiment.log_code("./lit_plankformer", include_fn=lambda path: path.endswith(".py"))
    else:
        logger = CSVLogger(save_dir=args.train_outpath, name='csv_logs')

    torch.backends.cudnn.allow_tf32 = False

    # Initialize the Model
    args.num_classes = len(datamodule.class_map)
    if args.balance_classes:
        args.class_weights = calculate_class_weights(datamodule)
    else:
        args.class_weights = None
    model = LitClassifier(**vars(args), finetune=True)  # TODO: check if this works on cscs, maybe add a file that downlaods model first
    model.load_datamodule(datamodule)

    # Initialize the Trainer
    trainer = l.Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=pl.callbacks.ModelCheckpoint(filename="best_model_acc_stage1", monitor="val_acc", mode="max"),
        check_val_every_n_epoch=args.max_epochs // 2,
        devices=gpus,
        strategy= "ddp" if gpus > 1 else "auto" ,
        enable_progress_bar=False,
        default_root_dir=args.train_outpath,
    )
    # Train the first and last layer of the model
    trainer.fit(model, datamodule=datamodule)
    # Load the best model from the first stage
    model = LitClassifier.load_from_checkpoint(str(trainer.checkpoint_callback.best_model_path), lr=args.lr * args.lr_factor, finetune=False)
    model.load_datamodule(datamodule)
    # sets up callbacks for stage 2
    callbacks = setup_callbacks(args.priority_classes, "best_model_acc_stage2")

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=2 * args.max_epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=max(args.max_epochs // 4,1),
        devices=gpus,
        strategy="ddp" if gpus > 1 else "auto",
        enable_progress_bar=False,
        default_root_dir=args.train_outpath,
    )
    trainer.fit(model, datamodule=datamodule)

    # Calculate and log the total time taken for training
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info("Time taken for training (in secs): {}".format(total_secs))
