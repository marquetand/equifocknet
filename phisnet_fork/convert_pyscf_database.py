"""
Script to convert PySCF orbital database to PhisNet format

(in minimal basis set for now)

PyScf outputs orbitals already in perfectly aligned order with atoms in DB.
Therefore the Fock matrix block 1x1 already corresponds to hydrogen 1 1s orbital interaction

So no need for tranformation back & forth
"""
import numpy as np
from ase.db import connect
import apsw
import os
import numpy as np
import argparse

from phisnet_fork.training.sqlite_database import HamiltonianDatabase
from phisnet_fork.utils.transform_hamiltonians import transform_hamiltonians_from_ao_to_lm
from phisnet_fork.utils.definitions import orbital_definitions

atom_type_list = ['', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']

def convert_pyscf_database_to_phisnet_format(read_db_file: str, write_db_file: str, orbital_definition: str, basis_set_size: int) -> None:
    read_db = connect(read_db_file)
    write_db = HamiltonianDatabase(write_db_file, flags=(apsw.SQLITE_OPEN_READWRITE | apsw.SQLITE_OPEN_CREATE))

    cursor = write_db._get_connection().cursor()

    for row in read_db.select(1):
        reference_atomic_numbers = row['numbers']
        write_db.add_Z(Z=reference_atomic_numbers)
    
    orbitals_reference = orbital_definitions[orbital_definition]
    atomic_symbols = []
    for Z in write_db.Z:
        write_db.add_orbitals(Z, orbitals_reference[Z])
        atomic_symbols.append(atom_type_list[Z])

    cursor.execute('''BEGIN''')
    for i, row in enumerate(read_db.select()):
        if i % 100 == 0:
            print(i)
        
        Z = row['numbers']
        assert np.all(Z == reference_atomic_numbers)

        R = row['positions'] * 1.8897261258369282 # convert angstrom to bohr
        H = transform_hamiltonians_from_ao_to_lm(row.data['F'].reshape(basis_set_size, basis_set_size), atoms=atomic_symbols, convention=orbital_definition)
        S = transform_hamiltonians_from_ao_to_lm(row.data['S'].reshape(basis_set_size, basis_set_size), atoms=atomic_symbols, convention=orbital_definition)
        write_db.add_data( R=R, E=None, F=None, H=H, S=S, C=None, transaction=False)
    cursor.execute('''COMMIT''')

if __name__ == "__main__":
    # example: python phisnet_fork/convert_pyscf_database.py --read_db_file fulvene_gs_2500_cc-pVDZ_molcas.db --orbital_definition fulvene_cc-pVDZ --basis_set_size 114
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_db_file', type=str)
    parser.add_argument('--orbital_definition', type=str)
    parser.add_argument('--basis_set_size', type=int)
    args = parser.parse_args()
    
    short = args.read_db_file.split('.')[0]
    read_db_file = f'./data_storage/{args.read_db_file}'
    write_db_file = f'./data_storage/{short}_phisnet_4.db'
    orbital_definition = args.orbital_definition
    basis_set_size = args.basis_set_size
    
    convert_pyscf_database_to_phisnet_format(
        read_db_file=read_db_file,
        write_db_file=write_db_file,
        orbital_definition=orbital_definition,
        basis_set_size=basis_set_size
    )