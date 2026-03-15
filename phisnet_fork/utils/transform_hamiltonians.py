import numpy as np
from typing import List
from argparse import Namespace


from phisnet_fork.utils.definitions import orbital_conventions, reverse_orbital_conventions

def transform_hamiltonians_from_ao_to_lm(hamiltonians: np.ndarray, atoms: List[str], convention: str) -> np.ndarray:
    """
    Transforms the ordering of elements in the hamiltonian from the AO ordering to the
    ordering of (l, m) as used in the spherical harmonics.
    """
    conv = orbital_conventions[convention]
    return transform_hamiltonians(hamiltonians, atoms, conv)

def transform_hamiltonians_from_lm_to_ao(hamiltonians: np.ndarray, atoms: List[str], convention: str) -> np.ndarray:
    """
    Transforms the ordering of elements in the hamiltonian from the AO ordering to the
    ordering of (l, m) as used in the spherical harmonics.
    """
    conv = reverse_orbital_conventions[convention]
    return transform_hamiltonians(hamiltonians, atoms, conv)

def transform_hamiltonians(hamiltonians: np.ndarray, atoms: List[str], conv: Namespace) -> np.ndarray:
    orbitals = ''
    orbitals_order = []
    for a in atoms:
        offset = len(orbitals_order)
        orbitals += conv.atom_to_orbitals_map[a]
        orbitals_order += [idx + offset for idx in conv.orbital_order_map[a]]

    transform_indices = []
    transform_signs = []
    for orb in orbitals:
        offset = sum(map(len, transform_indices))
        map_idx = conv.orbital_idx_map[orb]
        map_sign = conv.orbital_sign_map[orb]
        transform_indices.append((np.array(map_idx) + offset).tolist())
        transform_signs.append(map_sign)
    transform_indices = [item for sublist in transform_indices for item in sublist]
    transform_signs = [item for sublist in transform_signs for item in sublist]

    # transform_indices = [transform_indices[idx] for idx in orbitals_order]
    # transform_signs = [transform_signs[idx] for idx in orbitals_order]
    transform_indices = np.array(transform_indices).astype(int)
    transform_signs = np.array(transform_signs)

    hamiltonians_new = hamiltonians[...,transform_indices, :]
    hamiltonians_new = hamiltonians_new[...,:, transform_indices]
    hamiltonians_new = hamiltonians_new * transform_signs[:, None]
    hamiltonians_new = hamiltonians_new * transform_signs[None, :]

    return hamiltonians_new