###########
# IMPORTS #
###########

import argparse
import pathlib
import sys
from time import time
import numpy as np
import create_data
from models.setup_model import setup_model
from models.run_model import run_model
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
    # Loading Trained Input parameters
    # model_params=np.load("/scratch/snx3000/sdennis/eco_pomati/Plankiformer_OOD/trained_BEiT_models_Zoo/params.npy",allow_pickle=True).item()
    # if model_params.ttkind != 'image' and model_params.aug:
    #     model_params.aug = False
    # if model_params.ttkind == 'image':
    #     model_params.compute_extrafeat = 'no'
    # test_data = create_data.CreateDataset(args.test_path, args.L, args.resize_images, args.TTA)
    test_data=ZooplanktonDataModule(datapath=args.test_path, L=args.L, resize_images=args.resize_images, TTA=args.TTA, batch_size=args.batch_size)
    test_data.setup("test")
    # test_data.create_data_loaders()
    # initialize model training
    # combined_args={**vars(model_params),**vars(args)} #FIXME: order is important here
    # pprint.pprint(combined_args)
    # combined_args["last_layer_finetune"]=combined_args["last_layer_finetune"]=="yes"
    # combined_args["add_layer"]=combined_args["add_layer"]=="yes"
    model = setup_model(**vars(args))
    if torch.cuda.is_available() and args.use_gpu == 'yes':
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)
    # Do Predictions
    run_model(model, test_data=test_data,**vars(args)).run_ensemble_prediction_on_unseen( )
    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for prediction (in secs): {}'.format(total_secs))
