import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List, Dict, Any
from pymatgen.core.periodic_table import Element

from src.data.const import COVALENT_RADII
from src.models.loss.mic_dist import mic_pairwise_dist # TODO: Move to utils


class AtomicOverlapEnergy(nn.Module):
    def __init__(self, data: str, tolerance=0.8, threshold_mode: str = 'sum'):
        super().__init__()
        self.data = data  # dataset name: decides the cell orientation (src/evaluation/cell.py)
        self.tolerance = tolerance
        self.threshold_mode = threshold_mode
        
        radii = self._create_radius_tensor(COVALENT_RADII)
        self.register_buffer('radii_lookup', radii)
    
    def _create_radius_tensor(self, covalent_radii_dict: Dict[str, float]) -> Tensor:
        """Create a covalent radius lookup tensor indexed by atomic number."""
        radii = torch.zeros(119)
        for symbol, radius in covalent_radii_dict.items():
            try:
                z = Element(symbol).Z
                radii[z] = radius
            except:
                pass # Skip unknown symbols like "C_1"
        return radii
    
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
        # Compute pairwise distances with MIC
        dists = mic_pairwise_dist(cart_coords, lattice_params, self.data) # Shape: (B, N, N)
        
        # Look up Radii
        radii = self.radii_lookup[atomic_numbers] # (B, N)
        
        # Compute Thresholds 
        r_i = radii.unsqueeze(2)
        r_j = radii.unsqueeze(1)
        base = (r_i + r_j) if self.threshold_mode == 'sum' else torch.min(r_i, r_j)
        thresholds = base * self.tolerance # (B, N, N)
        
        # Mask invalid atoms and self-interactions
        pair_mask = (1 - torch.eye(atomic_numbers.shape[1], device=cart_coords.device)).unsqueeze(0)  # (1, N, N)
        pair_mask = pair_mask * (atom_mask[:, :, None] * atom_mask[:, None, :])  # (B, N, N)
        
        # Compute penalties
        violation = thresholds - dists
        active_violations = torch.relu(violation) * pair_mask
        
        # Energy Sum
        penalty_energy = torch.sum(active_violations**2, dim=(1, 2)) / 2.0
        
        return penalty_energy