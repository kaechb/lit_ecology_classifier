import copy

from time import time




import numpy as np


import torch

import torch.optim
import torch.utils.data
from scipy.stats import gmean




class run_model:
    def __init__(self,model,use_gpu, main_param_path,TTA, finetuned, test_data, add_layer, architecture,model_path, gpu_id, threshold, test_outpath,**kwargs):
        self.model=model
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        if torch.cuda.is_available() and use_gpu:
            torch.cuda.set_device(gpu_id)
            self.model.cuda(gpu_id)

        self.main_param_path = main_param_path
        self.TTA = TTA
        self.finetuned = "original" if finetuned==0 else "tuned" if finetuned==1 else "finetuned" if finetuned==2 else "finetuned"
        self.test_data = test_data
        self.add_layer = add_layer
        self.architecture = architecture
        self.model_path = model_path
        self.threshold = threshold
        self.test_outpath = test_outpath

        pass

    def cls_predict_on_unseen(self,test_dataloader):
        self.model.eval()
        outputs = []
        probs = []
        time_begin = time()
        with torch.no_grad():
            for i, (images) in enumerate(test_dataloader):

                images = images.to(self.device)
                output = self.model(images)
                if self.add_layer:
                    if self.architecture == 'deit' or self.architecture == 'vit' or self.architecture == 'mae':
                        output = torch.mean(output, 1)
                outputs.append(output)
                prob = torch.nn.functional.softmax(output, dim=1)
                probs.append(prob.cpu())

        total_secs = -1 if time_begin is None else (time() - time_begin)
        print('Time taken for prediction (in secs): {}'.format(total_secs))
        return torch.cat(probs)

    def run_ensemble_prediction_on_unseen(self):
        classes = np.load(self.main_param_path + '/classes.npy')
        Ensemble_prob = []
        im_names = self.test_data.filenames
        for i in range(len(self.model_path)):

            checkpoint_path = self.model_path[i]
            PATH = checkpoint_path + '/trained_model_' + self.finetuned + '.pth'
            checkpoint = torch.load(PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            dataloaders=self.test_data.test_dataloader()


            prob = self.cls_predict_on_unseen(  dataloaders[0])
            prob_1 = self.cls_predict_on_unseen(   dataloaders[1])
            prob_2 = self.cls_predict_on_unseen(   dataloaders[2])
            prob_3 = self.cls_predict_on_unseen(   dataloaders[3])

            Ensemble_prob.append(prob)
            Ensemble_prob.append(prob_1)
            Ensemble_prob.append(prob_2)
            Ensemble_prob.append(prob_3)
            probabilities = []
        Ens_DEIT_prob_max = []
        Ens_DEIT_label = []
        Ens_DEIT = []
        Ens_DEIT = gmean(Ensemble_prob)
            # for dl in dataloader:
            #     probs = self.cls_predict_on_unseen(dl)
            #     probabilities.append(probs)
            # ensemble_prob.append(torch.cat(probabilities).cpu())  # Concatenate all batch probabilities for this augmentation






        print(Ens_DEIT.shape)
        Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
        Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],
                                    dtype=object)
        Ens_DEIT_correcte