import pytorch_lightning as pl
import torch
from torchmetrics import MeanSquaredError

from phisnet_fork.utils.lr_scheduler import NoamLR

class PhisNet(pl.LightningModule):
  
  def __init__(self, model, args) -> None:
    super().__init__()
    
    self.model = model
    self.args = args
    
    self.loss_keys = ['full_hamiltonian', 'overlap_matrix']
    self.mse = MeanSquaredError()
    
  def forward(self, inputs):
    results = self.model(R=inputs['positions'])
    return results

  def save_model(self, model_path, do_postprocessing):
    torch.save(self.model, model_path)
  
  def loss_fn(self, pred, targets):
    loss = 0.0
    batch_size = pred[self.loss_keys[0]].shape[0]
    bs_size = pred[self.loss_keys[0]].shape[-1]
    for key in self.loss_keys:
      for i in range(batch_size):
         loss += torch.sum(torch.square(targets[key][i].flatten() - pred[key][i].flatten())) / (bs_size ** 2)
    return loss / (len(self.loss_keys) * batch_size)
  
  def log_metrics(self, targets, pred, flag):
    for key in self.loss_keys:
      metric = self.mse(pred[key], targets[key])
      self.log(
        f"{flag}_{key}_mse",
        metric,
        on_step=False,
        on_epoch=True,
        prog_bar=False
      )
  
  def training_step(self, batch, batch_idx):
      targets = batch
      pred = self(batch)
      loss = self.loss_fn(pred, targets)
      self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
      self.log_metrics(targets, pred, "train")
      return loss

  def validation_step(self, batch, batch_idx):
      torch.set_grad_enabled(True)
      targets = batch
      pred = self(batch)
      loss = self.loss_fn(pred, targets)
      self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
      self.log_metrics(targets, pred, "val")
      return {"val_loss": loss}

  def test_step(self, batch, batch_idx):
      targets = batch
      pred = self(batch)
      loss = self.loss_fn(pred, targets)
      self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
      self.log_metrics(targets, pred, "test")
      return {"test_loss": loss}

  
  def configure_optimizers(self):
    parameters = []
    weight_decay_parameters = []
    for name, param in self.model.named_parameters():
        if 'weight' in name and not 'radial_fn' in name and not 'embedding' in name:
            weight_decay_parameters.append(param)
        else:
            parameters.append(param)

    parameter_list = [
        {'params': parameters},
        {'params': weight_decay_parameters, 'weight_decay': float(self.args.weight_decay)}
    ]
  
    optimizer = torch.optim.AdamW(parameter_list,  lr=self.args.learning_rate, eps=self.args.epsilon, betas=(self.args.beta1, self.args.beta2), weight_decay=0.0, amsgrad=True)

    schedulers = []
    # schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.args.decay_factor, patience=self.args.decay_patience)
    schedule = NoamLR(optimizer, warmup_steps=15)
    optimconf = {"scheduler": schedule, "name": "lr_schedule", "monitor": "val_loss"}
    schedulers.append(optimconf)
    return [optimizer], schedulers
  

  