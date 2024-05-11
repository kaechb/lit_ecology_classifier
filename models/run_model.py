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
        ensemble_prob = []
        im_names = self.test_data.filenames
        for i in range(len(self.model_path)):

            checkpoint_path = self.model_path[i]
            PATH = checkpoint_path + '/trained_model_' + self.finetuned + '.pth'
            checkpoint = torch.load(PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])


            dataloader = self.test_data.test_dataloader()
            probabilities = []

            for dl in dataloader:
                probs = self.cls_predict_on_unseen(dl)
                probabilities.append(probs)
            ensemble_prob.append(torch.cat(probabilities).cpu())  # Concatenate all batch probabilities for this augmentation


        print(ensemble_prob)


        Ens_DEIT = gmean(ensemble_prob)
        print(Ens_DEIT.shape)
        Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)  # The class that the classifier would bet on
        Ens_DEIT_label = np.array([classes[Ens_DEIT_prob_max[i]] for i in range(len(Ens_DEIT_prob_max))],dtype=object)
        Ens_DEIT_label = Ens_DEIT_label.tolist()
        # Output the results
        self.output_results(im_names, Ens_DEIT_label)


    def output_results(self, im_names, labels):
        """Helper function to output prediction results."""
        name2 = 'geo_mean_'
        base_filename = f'{self.test_outpath}/Ensemble_models_Plankiformer_predictions_{name2}{self.finetuned}'
        print(f'I am using threshold value as : {self.threshold}' if self.threshold > 0 else 'I am using default value as threshold i.e. 0')

        label_set, suffix =labels,''
        file_path = f'{base_filename}{suffix}.txt'
        lines = [f'\n{img}------------------ {label}\n' for img, label in zip(im_names[0], label_set)]
        with open(file_path, 'w') as f:
            f.writelines(lines)  # Ensure using writelines for proper line-by-line writing



def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        return res


