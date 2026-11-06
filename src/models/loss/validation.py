import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher

from src.evaluation.cell import cell_matrix


def find_best_match_rmsd(args: tuple) -> list:
    """
    Computes the number of matching structures for a single batch item.
    This function is designed to be called by a multiprocessing worker.
    """
    (
        pred_coords_samples,        # (num_samples, num_atoms, 3)
        pred_lattices_samples,      # (num_samples, 6)
        true_coords,                # (num_atoms, 3)
        true_lattice,               # (6,)
        atom_types,                 # (num_atoms,)
        atom_mask,                  # (num_atoms,)
        block_type,                 # (num_atoms, )
        match_mode,                 # str: 'all' or 'framework'
        matcher_kwargs,
        data,                       # str: dataset name, decides the cell orientation
    ) = args

    # Determine mask
    if match_mode == 'framework':
        final_mask = atom_mask & (block_type != 3) # Exclude adsorbates
    else:
        final_mask = atom_mask

    matcher = StructureMatcher.from_dict(matcher_kwargs)
    
    true_structure = Structure(
        lattice=Lattice(cell_matrix(true_lattice, data)),
        species=atom_types[final_mask],
        coords=true_coords[final_mask],
        coords_are_cartesian=True,
    )

    rmsd_list = []
    for i in range(pred_coords_samples.shape[0]):
        try:
            pred_structure = Structure(
                lattice=Lattice(cell_matrix(pred_lattices_samples[i], data)),
                species=atom_types[final_mask],
                coords=pred_coords_samples[i, final_mask],
                coords_are_cartesian=True,
            )
            
            rmsd = matcher.get_rms_dist(true_structure, pred_structure)
            rmsd = rmsd if rmsd is None else rmsd[0]
            rmsd_list.append(rmsd)

        except (ValueError, np.linalg.LinAlgError):
            # Catch errors from invalid lattice parameters
            rmsd_list.append(None)
            continue

    return rmsd_list