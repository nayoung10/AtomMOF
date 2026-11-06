"""
Compute validity with MOFChecker
"""
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from src.utils.mofchecker_utils import is_mof_valid, EXPECTED_CHECK_VALUES
from src.evaluation.cell import cell_matrix, DATASETS

# Expected keys for MOF validity checks
EXPECTED_KEYS = list(EXPECTED_CHECK_VALUES.keys())


def compute_validity(pred_dict, num_samples, data, remove_solvent=True, use_gt=False) -> List[Dict[str, Any]]:
    """
    Computes validity for all samples in a single prediction dictionary.
    Returns a list of dictionaries with validity results for each sample.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Handle OOM exception
    if pred_dict is None:
        empty_result = {key: None for key in EXPECTED_KEYS}
        empty_result['is_valid'] = None
        empty_result['num_atoms'] = None
        return [empty_result for _ in range(num_samples)]
    
    # Create mask 
    if remove_solvent:
        mask = pred_dict['atom_block_type'] != 3 # Ignore solvent atoms
    else:
        mask = torch.ones_like(pred_dict['atom_types'], dtype=bool)

    num_atoms = int(mask.sum())
    
    results = []

    loop_range = range(1) if use_gt else range(num_samples)
    for k in loop_range:
        if use_gt:
            current_lattice = pred_dict['true_lattice']
            current_coords = pred_dict['true_coords']
        else:
            current_lattice = pred_dict['generated_lattice'][k]
            current_coords = pred_dict['generated_coords'][k]

        # Create ASE Atoms object
        atoms = Atoms(
            symbols=pred_dict['atom_types'][mask],
            positions=current_coords[mask],
            cell=cell_matrix(current_lattice, data),
            pbc=True
        )
        pred_structure = AseAtomsAdaptor.get_structure(atoms)

        # Commpute validity
        try:
            is_valid, desc = is_mof_valid(pred_structure)
            record: Dict[str, Any] = {key: desc.get(key, None) for key in EXPECTED_KEYS}
            record['is_valid'] = is_valid

        except Exception as e:
            record: Dict[str, Any] = {key: None for key in EXPECTED_KEYS}
            record['is_valid'] = None

        # Add sample index
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
    best_k_df = df_k.groupby('data_index')['is_valid'].max().reset_index()
    return best_k_df


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

    # Compute validity for all predictions
    delayed_tasks = [
        delayed(compute_validity)(pred_dict, args.num_samples, args.data, args.remove_solvent, args.use_gt)
        for pred_dict in prediction_list
    ] 
    results = Parallel(n_jobs=args.num_cpus, verbose=10)(tqdm(delayed_tasks, desc="Computing Validity")) # List[List[Dict]]

    # Flatten results
    flat_results = []
    for data_idx, sample_results in enumerate(results):
        for sample_dict in sample_results:
            sample_dict['data_index'] = data_idx
            flat_results.append(sample_dict)

    # Convert to DataFrame
    cols_order = ['data_index', 'sample_index', 'num_atoms', 'is_valid'] + EXPECTED_KEYS
    validity_df = pd.DataFrame(flat_results)[cols_order]

    # Save results as CSV
    name = f'validity_results_{args.data}.csv' if not args.use_gt else f'validity_results_gt_{args.data}.csv'
    output_file = pred_dir / name
    validity_df.to_csv(output_file, index=False)
    print(f"\nValidity results saved to {output_file}")

    # Print statistics
    for k in [1, 5, 10]:
        # Handle cases where num_samples < k
        if args.num_samples < k:
            continue
        
        best_k_df = get_best_k_results(validity_df, k)
        validity_rate = best_k_df['is_valid'].mean()
        print(f"Validity rate for best {k} samples: {validity_rate:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--data', type=str, required=True, choices=DATASETS)
    parser.add_argument('--keep_solvent', action='store_false', dest='remove_solvent') # default: solvent removed
    parser.add_argument('--use_gt', action='store_true') # default: use predictions
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_cpus', type=int, default=16) # NOTE: Do not set too high (zeopp crashes)
    args = parser.parse_args()
    main(args)