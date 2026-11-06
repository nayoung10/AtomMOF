import os
import json
import time
import argparse
from pathlib import Path
from functools import partial
from typing import List, Dict, Any, Optional

import torch
from p_tqdm import p_map
from ase import Atoms
from ase.io import write
from src.utils.zeopp_utils import mof_properties
from src.evaluation.cell import cell_matrix, DATASETS


def compute_properties(pred_dict: Dict, data_idx: int, zeo_dir: Path, data: str, remove_solvent: bool = True, sample_idx: int = 0, use_gt: bool = False) -> Optional[Dict]:
    # Handle OOM exception
    if pred_dict is None:
        return None

    # Create mask 
    if remove_solvent:
        mask = pred_dict['atom_block_type'] != 3 # Ignore solvent atoms
    else:
        mask = torch.ones_like(pred_dict['atom_types'], dtype=bool)

    # Construct ASE Atoms object
    if use_gt:
        lattice = pred_dict['true_lattice']
        coords = pred_dict['true_coords']
    else:
        lattice = pred_dict['generated_lattice'][sample_idx]
        coords = pred_dict['generated_coords'][sample_idx]

    # Create ASE Atoms objesct
    atoms = Atoms(
        symbols=pred_dict['atom_types'][mask],
        positions=coords[mask],
        cell=cell_matrix(lattice, data),
        pbc=True
    )

    # Create temporary CIF file
    temp_cif_path = zeo_dir / f"pred_{data_idx}_sample_{sample_idx}.cif"
    write(str(temp_cif_path), atoms)

    # Calculate ZEO++ properties
    try:
        results = mof_properties(
            cif_file=str(temp_cif_path), 
            zeo_store_path=str(zeo_dir), 
            cleanup=True, 
            save_file=False
        )

        return results

    except Exception as e:
        print(f"Error computing properties for data_idx {data_idx}, sample_idx {sample_idx}: {e}")
        return None

    finally: 
        # Clean up the temporary CIF file
        if temp_cif_path.exists():
            os.remove(temp_cif_path)


def main(args):
    # Set directory
    pred_dir = Path(args.pred_dir)

    # Get prediction file
    collated_file = pred_dir / 'collated_predictions.pt'
    prediction_list = torch.load(collated_file, map_location='cpu')
    
    # Create save directory
    zeo_dir = Path(args.pred_dir) / f"zeo_{args.data}"
    zeo_dir.mkdir(exist_ok=True)
    print(f"INFO:: Saving ZEO++ properties to {zeo_dir}")

    # Calculate ZEO++ properties
    start_time = time.time()

    compute_func = partial(
        compute_properties,
        zeo_dir=zeo_dir,
        data=args.data,
        remove_solvent=args.remove_solvent,
        use_gt=args.use_gt
    )

    zeo_props = p_map(
        compute_func, 
        prediction_list, 
        range(len(prediction_list)),
        num_cpus=args.num_cpus
    )

    elapsed_time = time.time() - start_time
    print(f"INFO:: ZEO++ properties calculated in {elapsed_time} seconds")

    # Save results
    suffix = "gt" if args.use_gt else "pred"

    with open(zeo_dir / f"time_zeo_{suffix}.txt", "w") as f:
        f.write(f"Time taken: {elapsed_time} seconds\n")

    with open(zeo_dir / f"zeo_props_relax_{suffix}.json", "w") as f:
        json.dump(zeo_props, f)

    print(f"INFO:: ZEO++ properties saved to {zeo_dir / f'zeo_props_relax_{suffix}.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str)
    parser.add_argument("--data", type=str, required=True, choices=DATASETS)
    parser.add_argument('--keep_solvent', action='store_false', dest='remove_solvent') # default: solvent removed
    parser.add_argument('--use_gt', action='store_true', dest='use_gt') # default: use predictions
    parser.add_argument("--num_cpus", type=int, default=48)
    args = parser.parse_args()
    main(args)