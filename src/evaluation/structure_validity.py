"""
Compute structure validity (distance matrix & volume check)
"""
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

from src.evaluation.cell import cell_matrix, DATASETS


def structure_validity(crystal: Structure, cutoff=0.5) -> bool:
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True

def compute_structure_validity(pred_dict, num_samples, data, cutoff=0.5, remove_solvent=True) -> List[Dict[str, Any]]:
    """
    Computes validity for all samples in a single prediction dictionary.
    Returns a list of dictionaries with validity results for each sample.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Handle OOM exception or missing data
    if pred_dict is None:
        return [{'is_valid': None, 'num_atoms': None, 'sample_index': k} 
                for k in range(num_samples)]
    
    # Create mask for atoms
    if remove_solvent:
        mask = pred_dict['atom_block_type'] != 3 # Ignore solvent atoms
    else:
        mask = torch.ones_like(pred_dict['atom_types'], dtype=bool)

    num_atoms = int(mask.sum())
    
    results = []

    for k in range(num_samples):
        pred_lattice = pred_dict['generated_lattice'][k]
        pred_coords = pred_dict['generated_coords'][k]
        pred_atom_types = pred_dict['atom_types']
        
        # Apply solvent mask if requested
        if remove_solvent:
            current_atom_types = pred_atom_types[mask]
            current_coords = pred_coords[mask]
        else:
            current_atom_types = pred_atom_types
            current_coords = pred_coords

        # Create ASE Atoms object
        try:
            pred_atoms = Atoms(
                symbols=current_atom_types,
                positions=current_coords,
                cell=cell_matrix(pred_lattice, data),
                pbc=True,
            )
            pred_structure = AseAtomsAdaptor.get_structure(pred_atoms)

            # Compute validity
            is_valid = structure_validity(pred_structure, cutoff=cutoff)
            
            # Record results
            record = {
                'is_valid': is_valid,
                'volume': pred_structure.volume, # recording volume for analysis
            }

        except Exception as e:
            # Catch geometric errors or conversion failures
            record = {
                'is_valid': None,
                'volume': None
            }

        # Add metadata
        record['sample_index'] = k
        record['num_atoms'] = num_atoms

        results.append(record)

    return results


def get_best_k_results(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Returns a Dataframe with one row per data index.
    Values are 1 (True) if ANY sample within the first k (0 to k-1) samples is valid.
    """
    df_k = df[df['sample_index'] < k]
    # We use .max() on boolean/int column: True/1 if at least one valid exists
    best_k_df = df_k.groupby('data_index')['is_valid'].max().reset_index()
    return best_k_df


def main(args):
    # Set directory
    pred_dir = Path(args.pred_dir)

    # Get prediction file
    collated_file = pred_dir / 'collated_predictions.pt'
    if not collated_file.exists():
        raise FileNotFoundError(f"Could not find {collated_file}")
        
    prediction_list = torch.load(collated_file, map_location='cpu')

    # Set num_samples
    if args.num_samples is None:
        # Find first non-None prediction to infer shape
        valid_pred = next((p for p in prediction_list if p is not None), None)
        if valid_pred is None:
            raise ValueError("All predictions are None due to OOM exceptions.")
        
        args.num_samples = valid_pred['generated_coords'].shape[0]
        print(f"Inferred num_samples: {args.num_samples}")

    # Compute validity for all predictions
    delayed_tasks = [
        delayed(compute_structure_validity)(
            pred_dict, 
            args.num_samples, 
            data=args.data,
            cutoff=args.cutoff, 
            remove_solvent=args.remove_solvent
        )
        for pred_dict in prediction_list
    ] 
    
    # Run in parallel
    results = Parallel(n_jobs=args.num_cpus, verbose=10)(
        tqdm(delayed_tasks, desc="Computing Structure Validity")
    ) # List[List[Dict]]

    # Flatten results
    flat_results = []
    for data_idx, sample_results in enumerate(results):
        for sample_dict in sample_results:
            sample_dict['data_index'] = data_idx
            flat_results.append(sample_dict)

    # Convert to DataFrame
    cols_order = ['data_index', 'sample_index', 'num_atoms', 'is_valid', 'volume']
    validity_df = pd.DataFrame(flat_results)[cols_order]

    # Save results as CSV
    output_file = pred_dir / f'structure_validity_results_{args.data}.csv'
    validity_df.to_csv(output_file, index=False)
    print(f"\nValidity results saved to {output_file}")

    # Print statistics
    print(f"\n--- Statistics (Cutoff: {args.cutoff}) ---")
    for k in [1, 5, 10]:
        # Handle case where we might have fewer samples than k
        if k > args.num_samples:
            continue
            
        best_k_df = get_best_k_results(validity_df, k)
        validity_rate = best_k_df['is_valid'].mean()
        print(f"Validity rate for best {k} samples: {validity_rate:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--data', type=str, required=True, choices=DATASETS)
    parser.add_argument('--keep_solvent', action='store_false', dest='remove_solvent')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--cutoff', type=float, default=0.5)
    parser.add_argument('--num_cpus', type=int, default=16)
    args = parser.parse_args()
    main(args)