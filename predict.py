###########
# IMPORTS #
###########

import argparse
import pathlib
import sys
from time import time
import numpy as np
import create_data
import setup_model
#from utils import prepare_data_for_testing as pdata_test
import logging
time_begin = time()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')






if __name__ == '__main__':
    print('\nRunning', sys.argv[0], sys.argv[1:])

    # Parse Arguments for prediction
    parser = argparse.ArgumentParser(description='Predict Classifier on Directory')

    parser.add_argument('--test_path', nargs='*', default=['./data/'], help="directory where you want to predict")
    parser.add_argument('--main_param_path', default='./trained_models/', help="main directory where the "
                                                                                    "training parameters are saved")
    parser.add_argument('--test_outpath', default='./preds/', help="directory where you want to save the predictions")

    parser.add_argument('--model_path', nargs='*',
                        default=['./out/trained_models/Init_0/',
                                    './out/trained_models/Init_1/'],
                        help='path of the saved models')
    parser.add_argument('--ensemble', type=int, default=0,
                        help="Set this to one if you want to ensemble multiple models else set it to zero")
    parser.add_argument('--finetuned', type=int, default=2, help='Choose "0" or "1" or "2" for finetuning')
    parser.add_argument('--threshold', type=float, default=0.0, help="Threshold to set")
    parser.add_argument('--resize_images', type=int, default=1,
                        help="Images are resized to a square of LxL pixels by keeping the initial image "
                                "proportions if resize=1. If resize=2, then the proportions are not kept but resized "
                                "to match the user defined dimension")
    parser.add_argument('--predict', type=int, default=1, help='Choose "0" for training and "1" for predicting')
    parser.add_argument('--use_gpu', choices=['yes', 'no'], default='no', help='Choose "no" to run using cpu')
    parser.add_argument('--gpu_id', type=int, default=0, help="select the gpu id ")
    parser.add_argument('--TTA', choices=['yes', 'no'], default='no',
                        help='Use test-time augmention or not')
    args = parser.parse_args()

    # Create Output directory
    pathlib.Path(args.test_outpath).mkdir(parents=True, exist_ok=True)
    logging.info('Loaded testing input parameters')
    # Loading Trained Input parameters

    model_params=np.load("/scratch/snx3000/sdennis/eco_pomati/Plankiformer_OOD/trained_BEiT_models_Zoo/params.npy",allow_pickle=True).item()
    for i, elem in enumerate(model_params.datapaths):
        model_params.datapaths[i] = elem + '/'
    model_params.outpath = model_params.outpath + '/'
    model_params.training_data = True if model_params.training_data == 'True' else False
    if model_params.ttkind != 'image' and model_params.aug == True:
        logging.warning('User asked for data augmentation, but we set it to False, because we only do it for `image` models')
        model_params.aug = False
    if model_params.ttkind == 'image':
        model_params.compute_extrafeat = 'no'
        logging.warning('User asked for computing extra features, but we set it to False, because we only do it for `mixed` models')
    logging.info("model parameters:{}".format(model_params))

    prep_test_data = create_data.CreateDataset(   args.test_path, model_params.L, model_params.compute_extrafeat, args.resize_images,
                         training_data=model_params.training_data)
    prep_test_data.CreateTrainTestSets(model_params, args)
    for_plankton_test = create_data.CreateDataForPlankton()
    for_plankton_test.make_test_set( args, prep_test_data)
    for_plankton_test.create_data_loaders(args)
    logging.info('init model')
    # initialize model training
    model_training = setup_model.import_and_train_model()
    logging.info('done with model')
    # Do Predictions
    model_training.load_model_and_run_prediction(model_params,args, for_plankton_test)

    total_secs = -1 if time_begin is None else (time() - time_begin)
    logging.info('Time taken for prediction (in secs): {}'.format(total_secs))
