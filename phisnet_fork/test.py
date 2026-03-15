from transform_hamiltonians import transform
import numpy as np

# water
# 1s, 2s, 2p, 1s, 1s

# H = np.multiply.outer(np.arange(7), np.arange(7))
# print(H)
# H_trans = transform(H, atoms=['O', 'H', 'H'], convention='orca_minimal')
# print(H_trans)


from pyscf import gto

molecule = gto.M(atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
                   basis = 'sto-3g',
                   spin=0,
                   symmetry=True)
print(molecule.ao_labels())

