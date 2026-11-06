"""
Rebuild a 3x3 cell from six lattice parameters (a, b, c, alpha, beta, gamma).

Six parameters do not fix the orientation of the cell, and stored Cartesian coordinates are only
meaningful with the cell they were written in:
  - BWDB preprocessing writes them with pymatgen's Lattice.from_parameters (c along z),
    src/preprocess/bwdb/extract_feats.py.
  - ODAC25 preprocessing writes them with ASE's cell (a along x), via ase.build.niggli_reduce.
The two orientations are identical only when alpha = beta = 90.

Reported by Tin Hadzi Veljkovic (CuspAI).
"""

import numpy as np
from ase.geometry import cellpar_to_cell
from pymatgen.core import Lattice

CELL_CONVENTION = {"bwdb": "pymatgen", "odac25": "ase"}
DATASETS = tuple(CELL_CONVENTION)


def cell_matrix(lattice_params, data: str) -> np.ndarray:
    """3x3 cell (rows = lattice vectors) in the orientation used by `data`'s preprocessing."""
    params = np.asarray(lattice_params, dtype=np.float64).reshape(6)
    if CELL_CONVENTION[data] == "pymatgen":
        return Lattice.from_parameters(*params).matrix
    return cellpar_to_cell(params)
