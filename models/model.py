import torch
import numpy as np
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage, Resize, RandomRotation, ToTensor
from scipy.stats import gmean
from time import time
from models.setup_model import setup_model

class Plankformer(LightningModule):
    def __init__(self, **hparms):
        super().__init__()
        self.save_hyperparameters()
        self.model = setup_model(**self.hparams)
        self.finetuned =  "original" if self.hparams.finetuned == 0 else "tuned" if self.hparams.finetuned == 1 else "finetuned" if self.hparams.finetuned == 2 else "finetuned"
        self.classes=torch.from_numpy(np.load(self.main_param_path + '/classes.npy'))

    def forward(self, x):
        if self.hparams.add_layer and (self.hparams.architecture in ['deit', 'vit', 'mae']):
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

    def run_ensemble_prediction_on_unseen(self):

        ensemble_prob = []

        for path in self.hparams.model_path:
            checkpoint_path = path + '/trained_model_' + self.finetuned + '.pth'
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            probabilities = []
            dataloader = self.test_data.test_dataloader()
            probabilities.extend(self.cls_predict_on_unseen(dataloader))
            ensemble_prob.append(torch.cat(probabilities).cpu())

        Ens_DEIT = gmean(ensemble_prob)
        Ens_DEIT_prob_max = Ens_DEIT.argmax(axis=1)
        Ens_DEIT_label = np.array([self.classes[idx] for idx in Ens_DEIT_prob_max], dtype=object)
        Ens_DEIT_label = Ens_DEIT_label.tolist()

        self.output_results(self.test_data.filenames, Ens_DEIT_label)

    def output_results(self, im_names, labels):
        name2 = 'geo_mean_'
        base_filename = f'{self.hparams.test_outpath}/Ensemble_models_Plankiformer_predictions_{name2}{self.finetuned}'
        file_path = f'{base_filename}.txt'
        lines = [f'{img}------------------ {label}\n' for img, label in zip(im_names, labels)]
        with open(file_path, 'w') as f:
            f.writelines(lines)

    def configure_optimizers(self):
        # Configure optimizers and learning rates here
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001 if self.hparams.finetuned== 0 else 1e-5)
        return optimizer
