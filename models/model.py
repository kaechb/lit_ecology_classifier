import torch
import numpy as np
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage, Resize, RandomRotation, ToTensor
from scipy.stats import gmean
from time import time

class ModelModule(LightningModule):
    def __init__(self, model, use_gpu, main_param_path, TTA, finetuned, test_data, add_layer, architecture, model_path, gpu_id, threshold, test_outpath, **kwargs):
        super().__init__()
        self.model = model
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.main_param_path = main_param_path
        self.TTA = TTA
        self.finetuned = "original" if finetuned == 0 else "tuned" if finetuned == 1 else "finetuned" if finetuned == 2 else "finetuned"
        self.test_data = test_data
        self.add_layer = add_layer
        self.architecture = architecture
        self.model_path = model_path
        self.threshold = threshold
        self.test_outpath = test_outpath
        self.gpu_id = gpu_id
        self.save_hyperparameters()

    def forward(self, x):
        if self.add_layer and (self.architecture in ['deit', 'vit', 'mae']):
            x = torch.mean(self.model(x), 1)
        else:
            x = self.model(x)
        return x

    def predict_step(self, batch, batch_idx):
        images = batch.to(self.device)
        logits = self(images)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs

    def test_step(self, batch, batch_idx):
        return self.predict_step(batch, batch_idx)

    def cls_predict_on_unseen(self, test_dataloader):
        probabilities = []
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                probs = self.predict_step(batch, None)
                probabilities.append(probs.cpu())
        return probabilities

    def run_ensemble_prediction_on_unseen(self):
        classes = np.load(self.main_param_path + '/classes.npy')
        ensemble_prob = []

        for path in self.model_path:
            checkpoint_path = path + '/trained_model_' + self.finetuned + '.pth'
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            probabilities = []
            dataloader = self.test_data.test_dataloader()
            probabilities.extend(self.cls_predict_on_unseen(dataloader))
            ensemble_prob.append(torch.cat(probabilities).cpu())

        Ens_DEIT = gmean(ensemble_prob)
        Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)
        Ens_DEIT_label = np.array([classes[idx] for idx in Ens_DEIT_prob_max], dtype=object)
        Ens_DEIT_label = Ens_DEIT_label.tolist()

        self.output_results(self.test_data.filenames, Ens_DEIT_label)

    def output_results(self, im_names, labels):
        name2 = 'geo_mean_'
        base_filename = f'{self.test_outpath}/Ensemble_models_Plankiformer_predictions_{name2}{self.finetuned}'
        file_path = f'{base_filename}.txt'
        lines = [f'{img}------------------ {label}\n' for img, label in zip(im_names, labels)]
        with open(file_path, 'w') as f:
            f.writelines(lines)

    def configure_optimizers(self):
        # Configure optimizers and learning rates here
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer
