import numpy as np
from typing import List, Tuple
from argparse import Namespace

from phisnet_fork.utils.definitions import orbital_conventions, reverse_orbital_conventions

def transform_hamiltonians_from_ao_to_lm(hamiltonians: np.ndarray, atoms: List[str], convention: str) -> np.ndarray:
    """
    Transforms the ordering of elements in the hamiltonian from the AO ordering to the
    ordering of (l, m) as used in the spherical harmonics.
    """
    conv = orbital_conventions[convention]
    return transform_hamiltonians(hamiltonians, atoms, conv, reverse=False)

def transform_hamiltonians_from_lm_to_ao(hamiltonians: np.ndarray, atoms: List[str], convention: str) -> np.ndarray:
    """
    Transforms the ordering of elements in the hamiltonian from the AO ordering to the
    ordering of (l, m) as used in the spherical harmonics.
    """
    conv = reverse_orbital_conventions[convention]
    return transform_hamiltonians(hamiltonians, atoms, conv, reverse=True)

def _permute_matrix(H: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Symmetric gather reorder of rows and columns."""
    return H[..., idx, :][..., :, idx]


def _build_global_indices(atoms: List[str], conv: Namespace) -> np.ndarray:
    """Per-orbital (de-)interleaving permutation, concatenated across atoms with offsets."""
    global_indices = []
    for a in atoms:
        offset = len(global_indices)
        global_indices += [idx + offset for idx in conv.global_order_map[a]]
    return np.array(global_indices, dtype=int)


def _build_within_shell_reorder(atoms: List[str], conv: Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Per-shell signed reorder. Each shell is permuted independently, so any number of
    shells of the same l (e.g. 3 p or 2 d) is handled automatically."""
    orbitals = ''.join(conv.atom_to_orbitals_map[a] for a in atoms)
    indices, signs, offset = [], [], 0
    for orb in orbitals:
        map_idx = conv.orbital_idx_map[orb]
        indices.extend(i + offset for i in map_idx)
        signs.extend(conv.orbital_sign_map[orb])
        offset += len(map_idx)
    return np.array(indices, dtype=int), np.array(signs)

def transform_hamiltonians(hamiltonians: np.ndarray, atoms: List[str], conv: Namespace, reverse: bool = False) -> np.ndarray:
    """Reorder a (batch of) hamiltonian(s) between AO and (l, m) ordering.

    Composes the global (de-)interleaving gather `g` (only if conv.ml_grouping) and the
    per-shell signed gather `t`/`s` into a single gather + sign multiply:
        forward:  c = g[t],  signs = s
        reverse:  c = t[g],  signs = s[g]   (g is the inverse stored in the reverse conv)
    """
    t, s = _build_within_shell_reorder(atoms, conv)

    if getattr(conv, 'ml_grouping', False):
        g = _build_global_indices(atoms, conv)
        indices, signs = (g[t], s) if not reverse else (t[g], s[g])
    else:
        indices, signs = t, s

    out = _permute_matrix(hamiltonians, indices)
    return out * signs[:, None] * signs[None, :]