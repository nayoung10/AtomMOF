import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from lightning.pytorch.utilities import rank_zero_info

from src.models.loss.mic_dist import mic_pairwise_dist
from src.models.energy import AtomicOverlapEnergy, MLIPEnergy
from src.models.simple.layers import AtomAttentionEncoder, AtomAttentionDecoder, TokenTransformer
from src.models.modules.utils import LinearNoBias, center_random_augmentation, default
from src.data.prior.prior_sampler import PriorSampler


class FlowModule(nn.Module):
    def __init__(
        self, 
        atom_s, 
        atom_z,
        token_s, 
        token_z,
        atom_feature_dim, 
        atom_encoder_depth, 
        atom_encoder_heads,
        atom_encoder_positional_encoding,
        token_transformer_depth, 
        token_transformer_heads, 
        atom_decoder_depth, 
        atom_decoder_heads,
        attention_impl,
        activation_checkpointing=False, 
        # **kwargs
    ):
        super().__init__()
        
        # Encoder
        self.atom_encoder = AtomAttentionEncoder(
            atom_s=atom_s, 
            atom_z=atom_z,
            atom_feature_dim=atom_feature_dim, 
            atom_encoder_depth=atom_encoder_depth, 
            atom_encoder_heads=atom_encoder_heads,
            positional_encoding=atom_encoder_positional_encoding,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing
        )
        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, token_s), nn.ReLU()
        )
        self.atom_to_token_trans_pair = nn.Sequential(
            LinearNoBias(atom_z, token_z), nn.ReLU()
        )

        # Transformer
        self.token_transformer = TokenTransformer(
            token_s=token_s, 
            token_z=token_z,
            token_transformer_depth=token_transformer_depth, 
            token_transformer_heads=token_transformer_heads, 
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing
        )
        self.token_to_atom_trans = nn.Sequential(
            LinearNoBias(token_s, atom_s), nn.ReLU()
        )

        # Decoder
        self.atom_decoder = AtomAttentionDecoder(
            atom_s=atom_s, 
            atom_decoder_depth=atom_decoder_depth, 
            atom_decoder_heads=atom_decoder_heads, 
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing
        )

    def _atom_to_token_single(self, atom_single, atom_to_token, multiplicity=1):
        atom_feats = self.atom_to_token_trans(atom_single)

        # Aggregate
        atom_to_token = atom_to_token.float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        token_feats = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_feats)
        return token_feats
    
    def _atom_to_token_pair(self, atom_pair, atom_to_token, multiplicity=1):
        atom_feats_pair = self.atom_to_token_trans_pair(atom_pair)
        # TODO: Aggregate for pair features
        return atom_feats_pair
    
    def _token_to_atom_single(self, token_single, atom_to_token, multiplicity=1):
        token_feats = self.token_to_atom_trans(token_single)

        # Broadcast
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_feats = torch.bmm(atom_to_token, token_feats)
        return atom_feats

    def forward(self, r_noisy, l_noisy, times, feats, multiplicity=1):
        # Encoder
        atom_single, atom_pair = self.atom_encoder(
            x_t=r_noisy, 
            l_t=l_noisy, 
            t=times, 
            feats=feats, 
            multiplicity=multiplicity
        )

        # Atom -> Token (# TODO: Add aggregation logic)
        token_single = self.atom_to_token_trans(atom_single)
        token_pair = self.atom_to_token_trans_pair(atom_pair)

        # Trunk
        token_single = self.token_transformer(
            x=token_single, 
            z=token_pair,
            t=times, 
            feats=feats, 
            multiplicity=multiplicity
        )

        # Token -> Atom (# TODO: Add broadcasting logic)
        atom_single = atom_single + self.token_to_atom_trans(token_single)

        # Decoder
        r_update, l_update = self.atom_decoder(
            x=atom_single, 
            t=times, 
            feats=feats, 
            multiplicity=multiplicity
        )

        return {"r_update": r_update, "l_update": l_update}


class AtomFlowMatching(Module):
    """Atom flow matching module"""
    NM_TO_ANG_SCALE = 10.0
    ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE
        
    def __init__(
        self, 
        flow_model_args: dict, 
        prior_sampler: PriorSampler, 
        num_sampling_steps: int = 16,
        coordinate_augmentation: bool = True,
        synchronize_timesteps: bool = False,
        compile_model: bool = False,
        **kwargs: dict
    ):
        super().__init__()

        # Model
        self.flow_model = FlowModule(**flow_model_args)
        if compile_model:
            # TODO: Check compile settings
            self.flow_model = torch.compile(self.flow_model, dynamic=False, fullgraph=False)

        # Priod
        self.prior_sampler = prior_sampler

        # Parameters
        self.num_sampling_steps = num_sampling_steps
        self.coordinate_augmentation = coordinate_augmentation
        self.synchronize_timesteps = synchronize_timesteps

    @property
    def device(self):
        return next(self.flow_model.parameters()).device

    def preconditioned_network_forward(
        self, 
        noised_atom_coords, 
        noised_lattice, 
        time, 
        network_condition_kwargs: dict, 
        training: bool = True
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(time, float):
            time = torch.full((batch,), time, device=device)

        # Scale inputs (TODO: properly scale each)
        r_noisy = self.ANG_TO_NM_SCALE * noised_atom_coords
        l_noisy = noised_lattice.clone()
        l_noisy[:, :3] = self.ANG_TO_NM_SCALE * noised_lattice[:, :3]
        l_noisy[:, 3:] = (torch.pi / 180.0) * noised_lattice[:, 3:]
        
        # Predict
        net_out = self.flow_model(
            r_noisy=r_noisy, 
            l_noisy=l_noisy, 
            times=time, 
            **network_condition_kwargs
        )

        # Rescale outputs
        denoised_coords = self.NM_TO_ANG_SCALE * net_out["r_update"]
        denoised_lattice = net_out["l_update"].clone()
        denoised_lattice[:, :3] = self.NM_TO_ANG_SCALE * denoised_lattice[:, :3]
        denoised_lattice[:, 3:] = (180.0 / torch.pi) * denoised_lattice[:, 3:]
        
        return denoised_coords, denoised_lattice

    def forward(self, feats, multiplicity=1, **kwargs):
        ############## Flow matching training step ##############

        batch_size = feats["cart_coords"].shape[0]

        # Sample timesteps
        if self.synchronize_timesteps:
            times = torch.rand(batch_size, device=self.device).repeat_interleave(multiplicity, 0)
        else:
            times = torch.rand(batch_size * multiplicity, device=self.device)

        # Prepare data
        atom_coords = feats["cart_coords"].repeat_interleave(multiplicity, 0)
        lattice = feats["lattice"].repeat_interleave(multiplicity, 0)
        atom_coords_0 = feats['cart_coords_0'].repeat_interleave(multiplicity, 0)
        lattice_0 = feats['lattice_0'].repeat_interleave(multiplicity, 0)
        
        # Add noise
        noised_atom_coords = (1 - times[:, None, None]) * atom_coords_0 + times[:, None, None] * atom_coords
        noised_lattice = (1 - times[:, None]) * lattice_0 + times[:, None] * lattice

        # Get model prediction
        denoised_atom_coords, denoised_lattice = self.preconditioned_network_forward(
            noised_atom_coords, 
            noised_lattice, 
            times, 
            training=True,
            network_condition_kwargs=dict(
                feats=feats, 
                multiplicity=multiplicity, 
                **kwargs
            ),
        )

        return dict(
            denoised_atom_coords=denoised_atom_coords, 
            denoised_lattice=denoised_lattice,
            times=times, 
            aligned_true_atom_coords=atom_coords, 
            aligned_true_lattice=lattice,
        )

    def compute_loss(
        self,
        feats,
        out_dict,
        multiplicity=1,
        loss_type: str='l2',
        use_dist_loss: bool=False,
        dist_cutoff: float=15.0,
        use_dist_schedule: bool=True
    ):
        if loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'l1':
            loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Must be 'l1' or 'l2'.")
        
        # Loss for Cartesian coordinates
        true_atom_coords = out_dict['aligned_true_atom_coords']
        pred_atom_coords = out_dict['denoised_atom_coords']
        atom_mask = feats["atom_pad_mask"].repeat_interleave(multiplicity, 0)

        coord_loss = loss_fn(pred_atom_coords, true_atom_coords, reduction='none')
        coord_loss = (coord_loss * atom_mask[..., None]).sum(dim=(1, 2)) / (atom_mask.sum(dim=1) + 1e-8)

        # Loss for lattice
        true_lattice = out_dict['aligned_true_lattice']
        pred_lattice = out_dict['denoised_lattice']

        length_loss = loss_fn(pred_lattice[:, :3], true_lattice[:, :3], reduction='none').mean(dim=1)
        angle_loss = loss_fn(pred_lattice[:, 3:], true_lattice[:, 3:], reduction='none').mean(dim=1)

        # Optional: MIC distance loss
        if use_dist_loss:
            # Compute MIC distances [B, N, N]
            data = getattr(self, 'data_name', None)  # set by SimpleMOFModule.setup from the datamodule
            if data is None:
                raise ValueError('use_dist_loss needs the dataset name to pick the cell orientation')
            true_dists = mic_pairwise_dist(true_atom_coords, true_lattice, data)
            pred_dists = mic_pairwise_dist(pred_atom_coords, pred_lattice, data)

            # Create mask
            pair_mask = (true_dists < dist_cutoff).float() # [B, N, N]
            pair_mask = pair_mask * (1 - torch.eye(true_dists.shape[1], device=true_dists.device).unsqueeze(0))
            pair_mask = pair_mask * (atom_mask[:, :, None] * atom_mask[:, None, :])

            # Compute loss
            dist_loss = loss_fn(pred_dists, true_dists, reduction='none')

            # Apply mask and average
            # Note: We sum over N x N grid and divide by number of valid pairs
            dist_loss = (dist_loss * pair_mask).sum(dim=(1, 2)) / (pair_mask.sum(dim=(1, 2)) + 1e-8)

            if use_dist_schedule:
                times = out_dict['times']
                t_weight = 1 + 8 * torch.relu(times - 0.5) # TODO: Try other schedules
                dist_loss = dist_loss * t_weight
        else:
            dist_loss = torch.zeros_like(coord_loss)

        return {
            # Per-sample losses of shape (batch_size * multiplicity,)
            'coord_loss': coord_loss,
            'dist_loss': dist_loss,
            'length_loss': length_loss,
            'angle_loss': angle_loss,
        }
    
    def get_score_from_velocity(self, vt, xt, t, atom_mask) -> torch.Tensor:
        """Convert velocity to score function.

        Args:
            vt: Velocity tensor of shape (B, N, 3)
            xt: Position tensor of shape (B, N, 3)
            t: Time scalar

        Returns:
            Score tensor of shape (B, N, 3)
        """
        vt = center_random_augmentation(vt, atom_mask, augmentation=False, centering=True)
        xt = center_random_augmentation(xt, atom_mask, augmentation=False, centering=True)
        score = (t * vt - xt) / (1 - t + 1e-6)
        return score
        
    @torch.no_grad()
    def _refine_step(
        self,
        atom_coords_t,
        lattice_t,
        atom_mask,
        multiplicity,
        network_condition_kwargs,
    ):
        """
        Performs a final refinement step by feeding the t=1 state through the model once more.
        """
        # Model's refined prediction at t=1
        pred_atom_coords_1, pred_lattice_1 = self.preconditioned_network_forward(
            noised_atom_coords=atom_coords_t,   # t=1 state from ODE
            noised_lattice=lattice_t,           # t=1 state from ODE
            time=1.0,
            training=False,
            network_condition_kwargs=dict(
                multiplicity=multiplicity,
                **network_condition_kwargs,
            )
        )

        final_atom_coords = pred_atom_coords_1
        final_lattice = pred_lattice_1

        # Ensure padding atoms remain at zero
        final_atom_coords = final_atom_coords * atom_mask.unsqueeze(-1)

        return final_atom_coords, final_lattice

    @torch.no_grad()
    def sample(
        self,
        atom_mask,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        use_sde_coords: bool = False,
        center_coords: bool = True,
        refine_final: bool = False,
        return_trajectory: bool = False,
        cached_priors: dict | None = None,
        steering_args: dict | None = None,
        # The following arguments are passed directly to the model
        **network_condition_kwargs,
    ):
        """
        Performs sampling from the learned flow ODE for both atom coordinates and lattice parameters.

        Starts with random noise at t=0 and integrates forward to t=1
        using a specified number of steps with the Euler method.

        Parameters
        ----------
        atom_mask : torch.Tensor
            A boolean tensor of shape (batch, num_atoms) indicating which atoms
            are real (True) and which are padding (False).
        num_sampling_steps : int, optional
            The number of discrete steps for the ODE solver. If None, uses
            the class default `self.num_sampling_steps`.
        multiplicity : int, optional
            The number of samples to generate per input, by default 1.
        center_coords : bool, optional
            If True, the coordinates will be centered after each sampling step. Defaults to True.
        refine_final : bool, optional
            If True, a final refinement step will be performed by passing the
            t=1 state through the model once more. Defaults to True.
        return_trajectory : bool, optional
            If True, the full coordinate and lattice trajectories from t=0 to t=1
            will be returned in the output dictionary. Defaults to False.
        **network_condition_kwargs :
            Additional keyword arguments passed directly to the flow model during
            the forward pass (e.g., s_inputs, s_trunk, z_trunk, etc.).

        Returns
        -------
        dict
            A dictionary containing the final generated atom coordinates and lattice
            under the keys "final_atom_coords" and "final_lattice". If `return_trajectory`
            is True, it will also contain the "trajectory" and "lattice_trajectory" keys.
        """
        ### (Optional) Setup steering configuration
        is_steering = steering_args is not None and steering_args.get("fk_steering", False)
        if is_steering:
            # Force use_sde_coords=True
            if not use_sde_coords:
                rank_zero_info("[Warning] FK steering is enabled but use_sde_coords is False. Setting use_sde_coords=True.")
                use_sde_coords = True

            # Default steering parameters
            num_particles = steering_args["num_particles"]
            fk_lambda = steering_args['fk_lambda']
            resampling_interval = steering_args["fk_resampling_interval"]
            fk_start_time = steering_args['fk_start_time']
            potential_mode = steering_args['potential_mode']
            energy_fn = steering_args['energy_fn']

            # Update multiplicity
            multiplicity = multiplicity * num_particles

            # Initialize energy trajectory ([batch * multiplicity, history])
            energy_traj = torch.empty((atom_mask.shape[0] * multiplicity, 0), device=self.device)

            # Initialize potential
            if energy_fn == "atomic_overlap":
                potential_fn = AtomicOverlapEnergy(
                    data=steering_args["data"],
                    tolerance=steering_args["tolerance"],
                ).to(self.device)
            elif energy_fn == "mlip":
                potential_fn = MLIPEnergy(data=steering_args['data'], device='cuda')
            else:
                raise ValueError(f"Unknown energy function: {energy_fn}")

            # Parepare atom_types for potential
            steering_atom_types = network_condition_kwargs.get("feats", {}).get("ref_element", None)
            steering_atom_types = steering_atom_types.repeat_interleave(multiplicity, 0) # Shape: [batch * multiplicity, num_atoms,]

        ### Standard setup and initialization
        num_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        batch_size = atom_mask.shape[0]
        num_atoms = atom_mask.shape[1]
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)

        # Dummy data dict for correct sampling shape
        sampler_data = {
            'cart_coords': torch.zeros((batch_size, num_atoms, 3), device=self.device),
            'atom_mask': atom_mask,
        }
        
        # Sample from prior
        if cached_priors is not None:
            priors = cached_priors
        else:
            priors = self.prior_sampler.sample(sampler_data)

        # Initialize coordinates and lattice from the prior sampler at t=0
        atom_coords_t = priors['cart_coords_0'] * atom_mask.unsqueeze(-1)
        lattice_t = priors['lattice_0']

        # Initialize lists to store trajectories if requested
        coord_trajectory = [atom_coords_t] if return_trajectory else None
        lattice_trajectory = [lattice_t] if return_trajectory else None

        ### ODE Solver Loop (Euler's Method)
        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # 1. Prediction (x_1 / x_0 prediction)
            pred_atom_coords_1, pred_lattice_1 = self.preconditioned_network_forward(
                noised_atom_coords=atom_coords_t,
                noised_lattice=lattice_t,
                time=t.item(),
                training=False,
                network_condition_kwargs=dict(
                    multiplicity=multiplicity,
                    **network_condition_kwargs,
                )
            )

            # ==================================================================
            # [FK STEERING] 2. Evaluate, Weight, and Resample
            # ==================================================================
            if is_steering and (t >= fk_start_time) and (i % resampling_interval == 0):
                # A. Compute energy of predict structure
                current_energy = potential_fn(
                    pred_atom_coords_1,
                    pred_lattice_1,
                    steering_atom_types,
                    atom_mask
                ) # Shape: [batch * multiplicity,]

                # B. Update trajectory
                energy_traj = torch.cat([energy_traj, current_energy.unsqueeze(1)], dim=1)

                # C. Compute log weights
                if potential_mode == "immediate":
                    # G_t = exp(-E_t / tau)
                    log_G = -1 * current_energy

                elif potential_mode == "difference":
                    # G_t = exp((E_{t-1} - E_t) / tau) -> Improvement is positive reward
                    if energy_traj.shape[1] == 1:
                        # No previous energy
                        log_G = torch.zeros_like(current_energy)
                    else:
                        # Improvement = Previous_Energy - Current_Energy
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                elif potential_mode == "max":
                    # G_t = exp(max_reward / tau) -> exp(-min_energy / tau)
                    best_energies = energy_traj.min(dim=1).values
                    log_G = -1 * best_energies

                elif potential_mode == "sum":
                    # G_t = exp(mean_reward / tau) -> exp(-mean_energy / tau)
                    # We average the trajectory to keep the scale consistent with tau
                    mean_energies = energy_traj.mean(dim=1)
                    log_G = -1 * mean_energies

                else:
                    raise ValueError(f"Unsupported potential_mode: {potential_mode}")

                # D. Resampling step
                # Resample within each original sample's particles
                # Reshape: [batch_size * original_multiplicity * num_particles, ] ->
                #          [batch_size * original_multiplicity, num_particles]
                log_G = log_G.reshape(-1, num_particles)
                weights = F.softmax(log_G * fk_lambda, dim=1)  # Shape: [batch_size * original_multiplicity, num_particles]

                # Sample indices (shape: [batch_size * original_multiplicity, num_particles])
                sampled_indices = torch.multinomial(weights, num_particles, replacement=True)

                # Adjust indices to flattened coordinates
                offset = torch.arange(weights.shape[0], device=self.device).unsqueeze(1) * num_particles
                flattened_indices = (sampled_indices + offset).flatten()

                # E. Apply selection
                atom_coords_t = atom_coords_t[flattened_indices]
                lattice_t = lattice_t[flattened_indices]
                pred_atom_coords_1 = pred_atom_coords_1[flattened_indices]
                pred_lattice_1 = pred_lattice_1[flattened_indices]
                energy_traj = energy_traj[flattened_indices]
            # ==================================================================

            # 3. Calculate the flow (vector field)
            # The flow is defined as u_t(x | x_1) = (x_1 - x) / (1 - t)
            flow_coords = (pred_atom_coords_1 - atom_coords_t) / (1 - t + 1e-5)
            flow_lattice = (pred_lattice_1 - lattice_t) / (1 - t + 1e-5)
            
            # Perform one step of Euler's method
            dt = t_next - t

            # 4. Step (Euler + SDE if specified)
            if use_sde_coords:
                # Define noise magnitude g^2(t)
                g2 = 0.5 * (1.0 - t)

                # Compute score and drift
                score_coords = self.get_score_from_velocity(
                    flow_coords, atom_coords_t, t.item(), atom_mask
                )
                drift = flow_coords + 0.5 * g2 * score_coords

                # Sample noise
                noise = torch.randn_like(atom_coords_t) * atom_mask.unsqueeze(-1)
                noise = center_random_augmentation(
                    noise, atom_mask, augmentation=False, centering=True
                )

                # Update coordinates
                atom_coords_t = atom_coords_t + drift * dt + torch.sqrt(g2 * dt) * noise
            else:
                atom_coords_t = atom_coords_t + flow_coords * dt
            
            lattice_t = lattice_t + flow_lattice * dt

            # Ensure padding atoms remain at zero
            atom_coords_t = atom_coords_t * atom_mask.unsqueeze(-1)

            # Center the coordinates
            atom_coords_t = center_random_augmentation(
                atom_coords_t, atom_mask, augmentation=False, centering=center_coords
            )

            # Store the current state if collecting trajectories
            if return_trajectory:
                coord_trajectory.append(atom_coords_t)
                lattice_trajectory.append(lattice_t)

        ### Final Refinement Step
        if refine_final:
            final_atom_coords, final_lattice = self._refine_step(
                atom_coords_t=atom_coords_t,
                lattice_t=lattice_t,
                atom_mask=atom_mask,
                multiplicity=multiplicity,
                network_condition_kwargs=network_condition_kwargs,
            )
        else:
            final_atom_coords = atom_coords_t
            final_lattice = lattice_t

        # Center the coordinates
        final_atom_coords = center_random_augmentation(
            final_atom_coords, atom_mask, augmentation=False, centering=center_coords
        )

        # Prepare the output dictionary
        output = {
            "sampled_atom_coords": final_atom_coords,   # (batch_size * multiplicity * num_particles, num_atoms, 3)
            "sampled_lattice": final_lattice,           # (batch_size * multiplicity * num_particles, 6)
        }

        # ==================================================================
        # [FK STEERING] Compute final energy
        # ==================================================================
        if is_steering:
            final_energy = potential_fn(
                final_atom_coords,
                final_lattice,
                steering_atom_types,
                atom_mask
            ) # Shape: [batch * multiplicity * num_particles,]
            output['final_energy'] = final_energy

            # Update trajectory
            energy_traj = torch.cat([energy_traj, final_energy.unsqueeze(1)], dim=1)
        # ==================================================================

        if return_trajectory:
            if refine_final:
                coord_trajectory.append(final_atom_coords)
                lattice_trajectory.append(final_lattice)

            # Stack the list of tensors into single trajectory tensors
            # If refine_step=True, shape is (num_steps + 2, batch_size * multiplicity * num_particles, ...)
            # If refine_step=False, shape is (num_steps + 1, batch_size * multiplicity * num_particles, ...)
            output["coord_trajectory"] = torch.stack(coord_trajectory, dim=0)
            output["lattice_trajectory"] = torch.stack(lattice_trajectory, dim=0)

            if is_steering:
                output["energy_trajectory"] = energy_traj # Shape: [batch * multiplicity * num_particles, history]
            
        return output
