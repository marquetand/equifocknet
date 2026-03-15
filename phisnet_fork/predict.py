from ase.db import connect
from pyscf import gto
from pyscf.tools import molden
import numpy as np

from training import *
from nn import *
from datamodule import CustomDataModule
from model import PhisNet
from train_new import load_model

import scipy
import scipy.linalg


def write_db_entry_to_molden_file(molden_file,
                                  atom_numbers,
                                  atom_positions,
                                  mo_coeffs=None,
                                  F=None):

    atom_string = ""
    for atom, position in zip(atom_numbers, atom_positions):
      atom_string += f'{atom} {position[0]} {position[1]} {position[2]}; '

    molecule = gto.M(atom=atom_string,
                    basis='sto_6g',
                    spin=0,
                    symmetry=True)
    
    if mo_coeffs is None:
      myscf = molecule.RHF()
      S = myscf.get_ovlp(molecule)
      mo_e, mo_coeffs = scipy.linalg.eigh(F, S)

    with open(molden_file, 'w') as f:
        molden.header(molecule, f)
        molden.orbital_coeff(molecule, f, mo_coeffs, ene=np.zeros(mo_coeffs.shape[0]))


def run_casscf_calculation(atom_numbers,
                           atom_positions,
                           guess_orbitals: np.ndarray,
                           basis='sto-6g'):
  
  atom_string = ""
  for atom, position in zip(atom_numbers, atom_positions):
    atom_string += f'{atom} {position[0]} {position[1]} {position[2]}; '

  molecule = gto.M(atom=atom_string,
                  basis='sto_6g',
                  spin=0,
                  symmetry=True)

  hartree_fock = molecule.RHF()
  n_states = 2
  weights = np.ones(n_states) / n_states
  casscf = hartree_fock.CASSCF(ncas=6, nelecas=6).state_average(weights)
  casscf.conv_tol = 1e-8

  e_tot, imacro, imicro, iinner, e_cas, ci, mo_coeffs, mo_energies = casscf.kernel(guess_orbitals)
  return imacro, imicro, iinner  


if __name__ == "__main__":
  args = parse_command_line_arguments()
  use_gpu = torch.cuda.is_available()
  if use_gpu:
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  
  datamodule = CustomDataModule(args)

  model = load_model(args, datamodule.dataset, use_gpu)
  phisnet = PhisNet(model=model, args=args)
  
  checkpoint = torch.load('checkpoints/ethene_hf_test-epoch=27-val_loss=0.24.ckpt')
  # checkpoint = torch.load('checkpoints/test_hf-epoch=94-val_loss=0.02.ckpt')
  phisnet.load_state_dict(checkpoint['state_dict'])
  phisnet.model.eval()
      
  with connect('ethene_s01.db') as conn:
    R = conn.get(1)['positions'] * 1.8897261258369282 # convert angstroms to bohr
  
    input = {'positions': torch.stack([torch.tensor(R, dtype=torch.float32)]).to(device)}
    output = phisnet(input)
    
    # F = output['full_hamiltonian'][0].detach().cpu().numpy()
    mo_coeffs = output['orbital_coefficients'][0].detach().cpu().numpy()
    # write_db_entry_to_molden_file('test.molden', conn.get(1)['numbers'], conn.get(1)['positions'], mo_coeffs=mo_coeffs)
    
    mo_coeffs = mo_coeffs.astype(np.double)
    results = run_casscf_calculation(conn.get(1)['numbers'], conn.get(1)['positions'], mo_coeffs)
    print(results)