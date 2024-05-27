import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import (DataLoader, Dataset, DistributedSampler,
                              random_split)

from ..data.tardataset import TarImageDataset


def custom_collate_fn(batch):
    batch_images = {rot: [] for rot in ["0",  "90",  "180", "270"]}
    batch_labels = []

    for rotated_images, label in batch:
        for rot in batch_images:
            batch_images[rot].append(rotated_images[rot])
        batch_labels.append(label)

    batch_images = {rot: torch.stack(batch_images[rot]) for rot in batch_images}
    batch_labels = torch.tensor(batch_labels)

    return batch_images, batch_labels


class PlanktonDataModule(LightningDataModule):
    def __init__(self, datapath, L=128, TTA=True, batch_size=32, dataset="",
                 use_data_moments=True,testing=False,calc_normalisation=False, random_rot=False,
                 AugMix=False, use_multi=True,ood=False, priority_classes=[], rest_classes=[],splits=[0.7,0.15],**kwargs):
        super().__init__()
        self.datapath = datapath
        self.TTA = TTA if testing else False
        self.batch_size = batch_size
        self.dataset = dataset
        self.use_data_moments = use_data_moments
        self.calc_normalisation = calc_normalisation
        self.random_rot = random_rot
        self.AugMix = AugMix
        self.use_multi = use_multi
        self.ood=ood
        self.priority_classes = priority_classes
        self.rest_classes = rest_classes
        self.train_split, self.val_split = splits

    def setup(self, stage=None):
        if stage == 'fit' or stage == "test":
            full_dataset = TarImageDataset(self.datapath, self.dataset, self.use_data_moments, random_rot=self.random_rot, TTA=self.TTA,train=True,calc_normalisation=self.calc_normalisation,AugMix=self.AugMix,ood=self.ood,priority_classes=self.priority_classes,rest_classes=self.rest_classes)
            train_size = int(self.train_split * len(full_dataset))
            val_size =  int(self.val_split * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [train_size, val_size,test_size],generator=torch.Generator().manual_seed(42))
            self.class_map = self.train_dataset.dataset.class_map
            self.val_dataset.train = False
            self.test_dataset.train = False


    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset) if torch.cuda.device_count() > 1 and self.use_multi else None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset) if torch.cuda.device_count() > 1  and self.use_multi else None
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=4, pin_memory=True, drop_last=False)
        if self.TTA:
            loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=4, pin_memory=True, drop_last=False, collate_fn=custom_collate_fn)
        return loader

    def test_dataloader(self):
        sampler = DistributedSampler(self.val_dataset) if torch.cuda.device_count() > 1  and self.use_multi else None
        if self.TTA:
            loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=4, pin_memory=True, drop_last=False, collate_fn=custom_collate_fn)
        else:
            loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,drop_last=False)
        return loader

if __name__ == '__main__':
    import os
    dm = PlanktonDataModule('/beegfs/desy/user/kaechben/eawag/training',dataset="phyto",use_data_moments=False)
    dm.setup('fit')
    test_loader = dm.train_dataloader()
    k=0
    for i in test_loader:
        print(i[0].shape,len(i[1]))
        for im in i[0]:
            import matplotlib.pyplot as plt
            plt.imshow(im.permute(1, 2, 0).numpy())
            plt.axis('off')  # Turn off axis
            plt.savefig(f"images/image_{k}.png", bbox_inches='tight', pad_inches=0)
            k+=1
        break
    print("number of images",k)