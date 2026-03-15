import pytorch_lightning
from pytorch_lightning.loggers import WandbLogger
import logging
import schnetpack as schnetpack

from phisnet_fork.utils.custom_data_module import CustomDataModule
from phisnet_fork.utils.phisnet import PhisNet
from phisnet_fork.training import *
from phisnet_fork.nn import *


def load_model(args, dataset, use_gpu):
  model =  NeuralNetwork(
        orbitals             = dataset.orbitals,
        order                = args.order,
        num_features         = args.num_features,
        num_basis_functions  = args.num_basis_functions,
        num_modules          = args.num_modules,
        num_residual_pre_x   = args.num_residual_pre_x,
        num_residual_post_x  = args.num_residual_post_x,
        num_residual_pre_vi  = args.num_residual_pre_vi,
        num_residual_pre_vj  = args.num_residual_pre_vj,
        num_residual_post_v  = args.num_residual_post_v,
        num_residual_output  = args.num_residual_output,
        num_residual_pc      = args.num_residual_pc,
        num_residual_pn      = args.num_residual_pn,
        num_residual_ii      = args.num_residual_ii,
        num_residual_ij      = args.num_residual_ij,
        num_residual_full_ii = args.num_residual_full_ii,
        num_residual_full_ij = args.num_residual_full_ij,
        num_residual_core_ii = args.num_residual_core_ii,
        num_residual_core_ij = args.num_residual_core_ij,
        num_residual_over_ij = args.num_residual_over_ij,
        basis_functions      = args.basis_functions,
        cutoff               = args.cutoff,
        activation           = args.activation)
  
  model.to(args.dtype)
  
  if use_gpu:
    model = model.cuda()
  
  return model

if __name__ == "__main__":
  args = parse_command_line_arguments()
  use_wandb = True
  use_gpu = torch.cuda.is_available()

  datamodule = CustomDataModule(args)

  model = load_model(args, datamodule.dataset, use_gpu)
  phisnet = PhisNet(model=model, args=args)
  phisnet.model.calculate_full_hamiltonian = True
  phisnet.model.calculate_core_hamiltonian = False
  phisnet.model.calculate_overlap_matrix = True
  phisnet.model.calculate_energy = True
  phisnet.model.calculate_forces = False
  
  callbacks = [
      schnetpack.train.ModelCheckpoint(
          monitor="val_loss",
          mode="min",
          save_top_k=1,
          save_last=True,
          dirpath="checkpoints",
          filename="{epoch:02d}",
          model_path="./checkpoints/" + args.model_name + ".pt"
      ),
      pytorch_lightning.callbacks.LearningRateMonitor(
        logging_interval="epoch"
      ),
      pytorch_lightning.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-6, 
        patience=50, 
        verbose=False, 
        mode="min"
      )
  ]
  
  if use_wandb:
    WAND_PROJECT = os.environ['WANDB_PROJECT']
    logger = WandbLogger(project=WAND_PROJECT)
    trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                        logger=logger,
                                        default_root_dir='./test/',
                                        max_epochs=1000,
                                        accelerator='gpu',
                                        devices=1)
  else:
    trainer = pytorch_lightning.Trainer(callbacks=callbacks, 
                                    default_root_dir='./test/',
                                    max_epochs=1000,
                                    accelerator='gpu',
                                    devices=1) 
  
  logging.info("Start training")
  trainer.fit(phisnet, datamodule=datamodule)