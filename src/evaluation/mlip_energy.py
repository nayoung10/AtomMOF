"""
Compute single-point energy (SPE) for both Predicted and Ground Truth structures using MLIP.
# TODO: Use multiple GPUs
"""
import os
import time
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal, Tuple
from dataclasses import dataclass

import torch
import pandas as pd
from ase import Atoms
from tqdm import tqdm

from src.evaluation.cell import cell_matrix, DATASETS


@dataclass
class EnergyResult:
    """Holds the results of single-point energy calculations for a data entry."""
    num_atoms: int
    energy_pred: Optional[float]
    energy_per_atom_pred: Optional[float]
    energy_gt: Optional[float]
    energy_per_atom_gt: Optional[float]
    success: bool
    error_message: Optional[str] = None


def load_calculator(
    model_name: str, 
    ckpt_dir: Optional[str] = None,
    device: Literal['cpu', 'cuda']='cuda'
):
    """Loads the specified MLIP model as a calculator."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit import load_predict_unit
    
    print(f"Loading MLIP model '{model_name}' on {device}...")
    
    if model_name == 'uma-s':
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
        calc = FAIRChemCalculator(predictor, task_name="odac")
    elif model_name == 'uma-m':
        predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=device)
        calc = FAIRChemCalculator(predictor, task_name="odac")
    elif model_name == 'esen':
        if not ckpt_dir:
            raise ValueError("mlip_ckpt_dir must be provided for 'esen' model.")
        ckpt_path = f"{ckpt_dir}/esen_sm_odac25_full.pt"
        predictor = load_predict_unit(ckpt_path, device=device)
        calc = FAIRChemCalculator(predictor)
    else:
        raise ValueError(f"Unknown MLIP model: {model_name}")

    return calc


def convert_entry_to_atoms(entry: Dict[str, Any], data: str, use_gt: bool = False, sample_idx: int = 0) -> Atoms:
    """Helper to convert dictionary entry to ASE Atoms object."""
    if use_gt:
        coords = entry['true_coords']
        lattice = entry['true_lattice']
    else:
        coords = entry['generated_coords'][sample_idx]
        lattice = entry['generated_lattice'][sample_idx]

    atoms = Atoms(
        symbols=entry['atom_types'],
        positions=coords,
        cell=cell_matrix(lattice, data),
        pbc=True
    )
    return atoms


def evaluate_pair(
    pred_dict: Dict[str, Any], 
    calculator: Any,
    data: str,
) -> EnergyResult:
    """
    Computes potential energy for both Ground Truth and Predicted structures.
    """
    try:
        # --- 1. Evaluate Predicted Structure ---
        atoms_pred = convert_entry_to_atoms(pred_dict, data, use_gt=False)
        atoms_pred.calc = calculator
        
        e_pred = atoms_pred.get_potential_energy()
        num_atoms = len(atoms_pred)
        epa_pred = e_pred / num_atoms

        # --- 2. Evaluate Ground Truth Structure ---
        atoms_gt = convert_entry_to_atoms(pred_dict, data, use_gt=True)
        atoms_gt.calc = calculator 
        
        e_gt = atoms_gt.get_potential_energy()
        epa_gt = e_gt / len(atoms_gt)

        return EnergyResult(
            num_atoms=num_atoms,
            energy_pred=e_pred,
            energy_per_atom_pred=epa_pred,
            energy_gt=e_gt,
            energy_per_atom_gt=epa_gt,
            success=True
        )

    except Exception as e:
        # In case of failure, return empty results with error message
        return EnergyResult(
            num_atoms=0,
            energy_pred=None,
            energy_per_atom_pred=None,
            energy_gt=None,
            energy_per_atom_gt=None,
            success=False,
            error_message=str(e)
        )


def setup_directories(pred_dir: Path, model_name: str, data: str) -> Tuple[Path, Path]:
    """Creates and returns the output and log directories."""
    output_dir = pred_dir / f'mlip_energy_{model_name}_{data}'
    logs_dir = output_dir / 'logs'

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving results to {output_dir}...")
    return output_dir, logs_dir


def save_results(
    results: List[EnergyResult],
    output_dir: Path,
    elapsed_time: float
):
    """Saves the energy results to CSV and logs time."""
    
    # Convert results to DataFrame
    data = []
    for res in results:
        data.append({
            'num_atoms': res.num_atoms,
            'energy_pred': res.energy_pred,
            'energy_per_atom_pred': res.energy_per_atom_pred,
            'energy_gt': res.energy_gt,
            'energy_per_atom_gt': res.energy_per_atom_gt,
            'success': res.success,
            'error': res.error_message
        })
    
    # Create DataFrame with specific column order
    cols = [
        'num_atoms', 'energy_pred', 'energy_per_atom_pred', 
        'energy_gt', 'energy_per_atom_gt', 'success', 'error'
    ]
    df = pd.DataFrame(data, columns=cols)
    df.index.name = 'data_index'
    
    csv_path = output_dir / 'energy_results.csv'
    df.to_csv(csv_path)
    print(f"Saved energy results to {csv_path}")

    # Save time log
    num_results = len(results)
    with open(output_dir / 'time_log.txt', 'w') as f:
        f.write(f"Total processing time (s): {elapsed_time:.2f}\n")
        if num_results > 0:
            f.write(f"Average time per structure pair (s): {elapsed_time / num_results:.2f}\n")


def main(args):
    pred_dir = Path(args.pred_dir)

    # 1. Load predictions
    collated_file = pred_dir / 'collated_predictions.pt'
    if not collated_file.exists():
        raise FileNotFoundError(f"Could not find {collated_file}")
        
    print(f"Loading predictions from {collated_file}...")
    prediction_list = torch.load(collated_file, map_location='cpu')

    # 2. Setup directories
    output_dir, logs_dir = setup_directories(pred_dir, args.mlip_model, args.data)

    # 3. Load Calculator
    calculator = load_calculator(
        args.mlip_model, 
        args.mlip_ckpt_dir, 
        args.mlip_device
    )

    print(f"Running energy evaluation for {len(prediction_list)} entries (Pred + GT)...")
    start_time = time.time()
    
    results: List[EnergyResult] = []
    
    # 4. Sequential Loop
    for i, pred_dict in tqdm(enumerate(prediction_list), total=len(prediction_list)):
        
        # Run evaluation
        result = evaluate_pair(pred_dict, calculator, args.data)
        
        # Log failures immediately
        if not result.success:
            with open(logs_dir / "failures.log", "a") as f:
                f.write(f"Index {i}: {result.error_message}\n")
        
        results.append(result)

    elapsed_time = time.time() - start_time
    print(f"Finished processing in {elapsed_time:.2f} seconds.")

    # 5. Save results
    save_results(results, output_dir, elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help="Directory containing collated_predictions.pt")
    parser.add_argument('--data', type=str, required=True, choices=DATASETS)
    parser.add_argument('--mlip_model', type=str, default='esen', choices=['uma-s', 'uma-m', 'esen'])
    parser.add_argument('--mlip_ckpt_dir', type=str, default=None, help="Required if using 'esen' model")
    parser.add_argument('--mlip_device', type=str, default='cuda')
    
    args = parser.parse_args()
    main(args)