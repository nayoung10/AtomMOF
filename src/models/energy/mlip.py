from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from ase import Atoms
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.datasets import data_list_collater
from fairchem.core.datasets.atomic_data import AtomicData

from src.evaluation.cell import cell_matrix

ESEN_CKPT_PATH = "logs/esen_sm_odac25_full.pt"


class MLIPEnergy(nn.Module):
    def __init__(self, data: str, device: str = "cuda"):
        super().__init__()
        self.data = data  # dataset name: decides the cell orientation (src/evaluation/cell.py)
        self.device = device
        self.predictor = load_predict_unit(ESEN_CKPT_PATH, device=device)

    def convert_to_atomic_data(self, cart_coords: Tensor, lattice_params: Tensor, atomic_numbers: Tensor, atom_mask: Tensor) -> List[AtomicData]:
        # Convert to numpy
        coords_np = cart_coords.detach().cpu().numpy()
        lattice_np = lattice_params.detach().cpu().numpy()
        atomic_numbers_np = atomic_numbers.detach().cpu().numpy()
        atom_mask_np = atom_mask.bool().detach().cpu().numpy()

        # Create ASE objects
        data_list = []
        batch_size = coords_np.shape[0]

        for i in range(batch_size):
            atoms = Atoms(
                numbers=atomic_numbers_np[i][atom_mask_np[i]],
                positions=coords_np[i][atom_mask_np[i]],
                cell=cell_matrix(lattice_np[i], self.data),
                pbc=True
            )
            data = AtomicData.from_ase(atoms, task_name='odac', r_edges=False, r_data_keys=["spin", "charge"])
            data_list.append(data)
            
        return data_list
    
    @torch.no_grad()
    def forward(self, cart_coords: Tensor, lattice_params: Tensor, atomic_numbers: Tensor, atom_mask: Tensor) -> Tensor:
        """
        Args:
            cart_coords: (B, N, 3) Cartesian coordinates (Angstroms)
            lattice_params: (B, 6) Lattice parameters
            atomic_numbers: (B, N) Atomic numbers (Int)
            atom_mask: (B, N) Mask for valid atoms
        Returns:
            penalty energy: (B,) Soft atomic overlap penalty energy
        """
        atomic_data_list = self.convert_to_atomic_data(cart_coords, lattice_params, atomic_numbers, atom_mask)
        batch = data_list_collater(atomic_data_list, otf_graph=True)

        out = self.predictor.predict(batch)
        energy = out['energy'].clone().detach()  # (B,)

        # Compute number of atoms with atomic numbers > 0
        num_atoms = atom_mask.sum(dim=1).float()  # (B,)

        # Normalize energy per atom
        energy = energy / num_atoms

        return energy