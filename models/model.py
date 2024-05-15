import torch
import numpy as np
from lightning import LightningModule
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage, Resize, RandomRotation, ToTensor
from scipy.stats import gmean
from time import time
from torch.optim.lr_scheduler import OneCycleLR
from models.setup_model import setup_model
def gmean(input_x, dim):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

class Plankformer(LightningModule):
    def __init__(self, **hparms):
        super().__init__()
        self.save_hyperparameters()
        self.model = setup_model(**self.hparams)
        self.classes=np.load(self.hparams.main_param_path + '/classes.npy')

    def forward(self, x):

            if self.hparams.add_layer and (self.hparams.architecture in ['deit', 'vit', 'mae']):
                x = torch.mean(self.model(x), 1)
            else:
                x = self.model(x)
            return x

    def load_datamodule(self, datamodule):
        self.dm = datamodule

    def predict_step(self, batch, batch_idx):
        images = batch.to(self.device)
        logits = self(images)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs

    def on_test_epoch_start(self) -> None:
        self.probabilities=[]
        self.filenames = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            probs = []


            if self.hparams.TTA:
                images, filenames = zip(*batch.values())
                images = torch.cat(images, dim=0)  # Combine all images into a single tensor
                filenames = filenames[0]  # Assume filenames are the same for all augmentations
            else:
                images, filenames = batch

            for m in self.model:
                m.to(self.device)
                m.eval()
                logits=m(images.to(self.device)).cpu()

                if self.hparams.TTA:
                    logits = torch.chunk(logits, 4, dim=0)
                    # Average predictions from the different augmentations
                    probs.extend([log.softmax( dim=1) for log in logits])
                else:
                    probs.append(torch.nn.functional.softmax(logits, dim=1))

            self.filenames.extend(filenames)
            self.probabilities.append(gmean(torch.stack(probs,dim=-1),dim=-1  ))

    def on_test_epoch_end(self) -> None:
        max_index = torch.cat(self.probabilities).argmax(axis=1)
        pred_label =np.array(self.classes[max_index.cpu().numpy()], dtype=object)
        self.output_results(self.filenames, pred_label)
        return super().on_test_epoch_end()

    def output_results(self, im_names, labels):
        name2 = 'geo_mean_'
        labels = labels.tolist()
        base_filename = f'{self.hparams.test_outpath}/Ensemble_models_Plankiformer_predictions_{name2}{self.hparams.finetuned}'
        file_path = f'{base_filename}.txt'
        lines = [f'\n{img}------------------ {label}\n' for img, label in zip(im_names, labels)]
        with open(file_path, 'w') as f:
            f.writelines(lines)

    def configure_optimizers(self):
        # Configure optimizers and learning rates here
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001 if self.hparams.finetuned== "original" else 1e-5)
        scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(self.train_dataloader()), epochs=self.trainer.max_epochs)
        return [optimizer], [scheduler]
