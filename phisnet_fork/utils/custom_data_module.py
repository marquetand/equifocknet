import pytorch_lightning as pl
import os

from phisnet_fork.training import *

class CustomDataModule(pl.LightningDataModule):
  
  def __init__(self, args):
    super().__init__()
    dataset_path = os.path.join('/home/rhjvanworkum/caschnet/', f'data_storage/{args.dataset}')
    split_path = os.path.join('/home/rhjvanworkum/caschnet/', f'data_storage/{args.split_file}')
    
    self.dataset = HamiltonianDataset(dataset_path, dtype=args.dtype)
    self._train_dataset, self._valid_dataset, self._test_dataset = dataset_split_by_file(self.dataset, split_path)
    self.args = args
    self.n_workers = args.num_workers
    
  
  @property
  def train_dataset(self):
      return self._train_dataset

  @property
  def val_dataset(self):
      return self._valid_dataset

  @property
  def test_dataset(self):
      return self._test_dataset
    
  def train_dataloader(self):
      return torch.utils.data.DataLoader(
          self._train_dataset,
          batch_size=self.args.train_batch_size,
          num_workers=self.n_workers,
          shuffle=True,
          pin_memory=True,
          collate_fn=lambda batch: self.dataset.collate_fn(batch)
      )

  def val_dataloader(self):
      return torch.utils.data.DataLoader(
          self._valid_dataset,
          batch_size=self.args.valid_batch_size,
          num_workers=self.n_workers,
          pin_memory=True,
          collate_fn=lambda batch: self.dataset.collate_fn(batch)
      )


  def test_dataloader(self):
      return torch.utils.data.DataLoader(
          self._test_dataset,
          batch_size=self.args.valid_batch_size,
          num_workers=self.n_workers,
          pin_memory=True,
          collate_fn=lambda batch: self.dataset.collate_fn(batch)
      )
