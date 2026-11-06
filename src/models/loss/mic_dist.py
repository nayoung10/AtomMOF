import torch
import numpy as np


def lattice_params_to_matrix(params: torch.Tensor) -> torch.Tensor:
    """
    Converts lattice parameters to a 3x3 lattice matrix (differentiable).
    
    Args:
        params: Tensor of shape [batch_size, 6]
                Columns must be: (a, b, c, alpha, beta, gamma)
                Angles (alpha, beta, gamma) are assumed to be in DEGREES.
    
    Returns:
        lattice_matrix: Tensor of shape [batch_size, 3, 3]
                        Rows are lattice vectors a, b, c.
    """
    # 1. Unpack parameters
    # shape: [batch_size]
    a = params[:, 0]
    b = params[:, 1]
    c = params[:, 2]
    alpha_deg = params[:, 3]
    beta_deg  = params[:, 4]
    gamma_deg = params[:, 5]

    # 2. Convert angles to radians
    deg_to_rad = np.pi / 180.0
    alpha = alpha_deg * deg_to_rad
    beta  = beta_deg  * deg_to_rad
    gamma = gamma_deg * deg_to_rad

    # 3. Pre-compute trigonometric values
    cos_alpha = torch.cos(alpha)
    cos_beta  = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    # 4. Construct Lattice Vectors (following the standard convention)
    
    # Vector a: Aligned with x-axis
    # a_vec = [a, 0, 0]
    zeros = torch.zeros_like(a)
    val_a_x = a
    val_a_y = zeros
    val_a_z = zeros

    # Vector b: In xy-plane
    # b_vec = [b * cos(gamma), b * sin(gamma), 0]
    val_b_x = b * cos_gamma
    val_b_y = b * sin_gamma
    val_b_z = zeros

    # Vector c: General position
    # Projection math ensures dot products match angles
    val_c_x = c * cos_beta
    val_c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    
    # c_z is the remainder of the length c
    # c_z = sqrt(c^2 - c_x^2 - c_y^2)
    val_c_z = torch.sqrt(c**2 - val_c_x**2 - val_c_y**2 + 1e-6) 
    # (+ 1e-6 added for numerical stability in sqrt gradient)

    # 5. Stack into a matrix
    # We stack along dim=1 to create rows.
    # Result shape: [batch_size, 3, 3]
    
    row_a = torch.stack([val_a_x, val_a_y, val_a_z], dim=1)
    row_b = torch.stack([val_b_x, val_b_y, val_b_z], dim=1)
    row_c = torch.stack([val_c_x, val_c_y, val_c_z], dim=1)
    
    lattice_matrix = torch.stack([row_a, row_b, row_c], dim=1)
    
    return lattice_matrix


def lattice_params_to_matrix_pymatgen(params: torch.Tensor) -> torch.Tensor:
    """Same as pymatgen Lattice.from_parameters (c along z, a in the xz plane), differentiable.

    params: [B, 6] = (a, b, c, alpha, beta, gamma), angles in degrees. Returns [B, 3, 3], rows = a, b, c.
    """
    a, b, c = params[:, 0], params[:, 1], params[:, 2]
    alpha, beta, gamma = (params[:, 3:6] * (np.pi / 180.0)).unbind(dim=1)
    val = (torch.cos(alpha) * torch.cos(beta) - torch.cos(gamma)) / (torch.sin(alpha) * torch.sin(beta))
    gamma_star = torch.arccos(val.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    zeros = torch.zeros_like(a)
    row_a = torch.stack([a * torch.sin(beta), zeros, a * torch.cos(beta)], dim=1)
    row_b = torch.stack(
        [-b * torch.sin(alpha) * torch.cos(gamma_star), b * torch.sin(alpha) * torch.sin(gamma_star), b * torch.cos(alpha)],
        dim=1,
    )
    row_c = torch.stack([zeros, zeros, c], dim=1)
    return torch.stack([row_a, row_b, row_c], dim=1)


def lattice_params_to_matrix_for(params: torch.Tensor, data: str) -> torch.Tensor:
    """Cell matrix in the orientation used by `data`'s preprocessing (see src/evaluation/cell.py)."""
    from src.evaluation.cell import CELL_CONVENTION
    if CELL_CONVENTION[data] == "pymatgen":
        return lattice_params_to_matrix_pymatgen(params)
    return lattice_params_to_matrix(params)


def mic_pairwise_dist(cart_coords: torch.Tensor, lattice_params: torch.Tensor, data: str) -> torch.Tensor:
    """
    Computes MIC distances robustly for ANY lattice (Triclinic, Hexagonal, etc.)
    by checking the 27 nearest periodic images.
    
    Args:
        cart_coords: [batch, num_atoms, 3]
        lattice_params: [batch, 6] (a, b, c, alpha, beta, gamma)
    """    
    # 1. Get Lattice Matrix and Inverse
    lattice_matrix = lattice_params_to_matrix_for(lattice_params, data) # [B, 3, 3]
    inv_lattice = torch.inverse(lattice_matrix) # [B, 3, 3]

    # 2. Convert to Fractional Coordinates
    frac_coords = torch.bmm(cart_coords, inv_lattice) 

    # 3. Compute Pairwise Fractional Differences
    # Shape: [B, N, N, 3]
    diff_frac = frac_coords[:, :, None, :] - frac_coords[:, None, :, :]
    
    # 4. Apply Initial Wrap (roughly centers the diffs near 0)
    diff_frac = diff_frac - torch.round(diff_frac)

    # 5. Generate Neighbor Offsets (The 27 images)
    shifts = torch.tensor([
        [i, j, k] 
        for i in (-1, 0, 1) 
        for j in (-1, 0, 1) 
        for k in (-1, 0, 1)
    ], device=cart_coords.device, dtype=cart_coords.dtype) # Shape [27, 3]

    # 6. Broadcast and Compute All 27 Cartesian Distances
    # diff_frac shape: [B, N, N, 3]
    # shifts shape:    [27, 3]
    
    # Reshape for broadcasting:
    # diff_frac: [B, N, N, 1,  3]
    # shifts:    [1, 1, 1, 27, 3]
    diff_frac_expanded = diff_frac.unsqueeze(3)
    shifts_expanded = shifts.view(1, 1, 1, 27, 3)
    
    # All candidate fractional vectors: [B, N, N, 27, 3]
    candidate_frac = diff_frac_expanded + shifts_expanded
    
    # Convert all candidates to Cartesian: [B, N, N, 27, 3]
    # We multiply by lattice matrix.
    candidate_cart = torch.einsum('bnmkx,bxy->bnmky', candidate_frac, lattice_matrix)

    # 7. Compute Distances and Take the Minimum
    candidate_dists = torch.norm(candidate_cart, dim=-1)
    
    # Min over the 27 images -> [B, N, N]
    true_mic_dists, _ = torch.min(candidate_dists, dim=-1)
    
    return true_mic_dists