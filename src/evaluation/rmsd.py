"""
Compute match rate and RMSD between predicted and true structures.
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

from src.evaluation.cell import cell_matrix, DATASETS


def compute_rmsd(pred_dict, num_samples, data, remove_solvent=True, stol=0.5, ltol=0.3, angle_tol=10) -> Tuple[List[float], int]:
    """
    Computes RMSD for all samples in a single prediction dictionary.
    """
    # Handle OOM exception
    if pred_dict is None:
        return [np.nan] * num_samples, 0
    
    # Create mask
    if remove_solvent:
        mask = pred_dict['atom_block_type'] != 3 # Ignore solvent atoms
    else:
        mask = torch.ones_like(pred_dict['atom_types'], dtype=bool)

    num_atoms = int(mask.sum())
    
    # Initialize StructureMatcher
    matcher = StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol)
    rmsd_list = []

    for k in range(num_samples):
        # Select predicted structure
        pred_lattice = pred_dict['generated_lattice'][k]
        pred_coords = pred_dict['generated_coords'][k]

        # Create ASE Atoms object
        pred_atoms = Atoms(
            symbols=pred_dict['atom_types'][mask],
            positions=pred_coords[mask],
            cell=cell_matrix(pred_lattice, data),
            pbc=True
        )

        true_atoms = Atoms(
            symbols=pred_dict['atom_types'][mask],
            positions=pred_dict['true_coords'][mask],
            cell=cell_matrix(pred_dict['true_lattice'], data),
            pbc=True
        )

        # Convert to pymatgen structures
        pred_structure = AseAtomsAdaptor.get_structure(pred_atoms)
        true_structure = AseAtomsAdaptor.get_structure(true_atoms)

        # Match structures
        rmsd = matcher.get_rms_dist(pred_structure, true_structure)
        rmsd = rmsd[0] if rmsd is not None else np.nan
        rmsd_list.append(rmsd)

    return rmsd_list, num_atoms


def main(args):
    # Set directory
    pred_dir = Path(args.pred_dir)

    # Get prediction file
    collated_file = pred_dir / 'collated_predictions.pt'
    prediction_list = torch.load(collated_file, map_location='cpu')

    # Set num_samples
    if args.num_samples is None:
        valid_pred = next((p for p in prediction_list if p is not None), None)
        if valid_pred is None:
            raise ValueError("All predictions are None due to OOM exceptions.")
        
        args.num_samples = valid_pred['generated_coords'].shape[0]
        print(f"Inferred num_samples: {args.num_samples}")

    # Compute RMSD and match rate
    print(f"Starting RMSD computation on {len(prediction_list)} samples...")

    # Prepare for parallel processing
    delayed_tasks = [
        delayed(compute_rmsd)(
            pred_dict,
            num_samples=args.num_samples,
            data=args.data,
            remove_solvent=args.remove_solvent,
            stol=args.stol,
            ltol=args.ltol,
            angle_tol=args.angle_tol
        )
        for pred_dict in prediction_list
    ] 
    results = Parallel(n_jobs=args.num_cpus)(tqdm(delayed_tasks, desc="Computing RMSD"))
    rmsd_data, num_atoms_data = zip(*results) # Unzip results

    # Convert to DataFrame
    rmsd_df = pd.DataFrame(list(rmsd_data), columns=[k for k in range(args.num_samples)])
    rmsd_df.index.name = 'data_index'
    rmsd_df['best'] = rmsd_df.min(axis=1)
    rmsd_df['num_atoms'] = list(num_atoms_data)

    # Save results as CSV
    output_file = pred_dir / f'rmsd_results_{args.data}_stol-{args.stol}_ltol-{args.ltol}_angle-{args.angle_tol}.csv'
    rmsd_df.to_csv(output_file, index=True)
    print(f"\nRMSD results saved to {output_file}")

    # Print statistics
    print("\nSummary Statistics:")
    match_rate = (len(rmsd_df) - rmsd_df.isna().sum(axis=0)) / len(rmsd_df)
    avg_rmsd = rmsd_df.mean(axis=0)
    print(f"Match rate:\n {match_rate}")
    print(f"\nAverage RMSD:\n{avg_rmsd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--data', type=str, required=True, choices=DATASETS)
    parser.add_argument('--keep_solvent', action='store_false', dest='remove_solvent')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_cpus', type=int, default=1) # NOTE: 1 is faster
    parser.add_argument('--stol', type=float, default=0.5)
    parser.add_argument('--ltol', type=float, default=0.3)
    parser.add_argument('--angle_tol', type=float, default=10)
    args = parser.parse_args()
    main(args)