###########
# IMPORTS #
###########

import argparse
import pathlib
import sys
import lightning
from time import time
import numpy as np
import create_data
from models.setup_model import setup_model
from models.run_model import run_model
from models.model import Plankformer
from data.datamodule import ZooplanktonDataModule
from helpers.argparser import argparser

import torch
#from utils import prepare_data_for_testing as pdata_test
import logging
time_begin = time()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    print('\nRunning', sys.argv[0], sys.argv[1:])

    # Parse Arguments for prediction
    parser = argparser()
    args = parser.parse_args()
    # Create Output directory
    pathlib.Path(args.test_outpath).mkdir(parents=True, exist_ok=True)

    data_module=ZooplanktonDataModule(datapath=args.test_path, L=args.L, resize_images=args.resize_images, TTA=args.TTA, batch_size=args.batch_size)
    data_module.setup("test")
    model = Plankformer(**vars(args))
    model.load_datamodule(data_module)
    if torch.cuda.is_available() and args.use_gpu == 'yes':
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)
    # Do Predictions
    trainer = lightning.Trainer()
    trainer.test(model,datamodule=data_module )
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for prediction (in secs): {}'.format(total_secs))
