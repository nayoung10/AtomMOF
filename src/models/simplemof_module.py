import os
import gc
import time
import random
import multiprocessing as mp
from typing import Any, Optional

import torch
from einops import rearrange
from lightning import Callback, LightningModule
from lightning.pytorch.utilities import rank_zero_info
from torch import Tensor, nn
from torchmetrics import MeanMetric
from pebble import ProcessPool
from concurrent.futures import TimeoutError as PebbleTimeoutError
from pymatgen.analysis.structure_matcher import StructureMatcher

from src.models.loss.validation import find_best_match_rmsd
from src.models.loss.utils import stratify_loss_by_time
from src.models.simple.flow import AtomFlowMatching
from src.models.optim.ema import EMA
from src.models.optim.scheduler import AlphaFoldLRScheduler


class SimpleMOFModule(LightningModule):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atom_feature_dim: int,
        flow_model_args: dict[str, Any],
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        flow_process_args: dict[str, Any],
        ema: bool = False,
        ema_decay: float = 0.999,
        predict_args: Optional[dict[str, Any]] = None,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # EMA
        self.use_ema = ema
        self.ema_decay = ema_decay

        # Args
        self.training_args = training_args
        self.validation_args = validation_args
        self.predict_args = predict_args

        # Kernels
        self.use_kernels = use_kernels

        # Validation metrics
        self.matcher_kwargs = StructureMatcher(**self.validation_args['structure_matcher_args']).as_dict()
        self.val_match_rate = MeanMetric(sync_on_compute=False)
        self.val_rmsd = MeanMetric(sync_on_compute=False)
        self.validation_step_outputs = []

        # Flow matching module
        self.structure_module = AtomFlowMatching(
            flow_model_args={
                "atom_s": atom_s,
                "atom_z": atom_z,
                "token_s": token_s,
                "token_z": token_z,
                "atom_feature_dim": atom_feature_dim,
                **flow_model_args,
            },
            **flow_process_args,
        )
    
    def _data_name(self) -> str:
        """Dataset name (bwdb / odac25) from the datamodule; decides the cell orientation (src/evaluation/cell.py)."""
        dm = getattr(self.trainer, "datamodule", None) if self._trainer is not None else None
        name = getattr(dm, "lmdb_suffix", None)
        if name is None:
            raise ValueError("Cannot determine the dataset name (datamodule.lmdb_suffix) for the cell orientation")
        return name

    def setup(self, stage: str) -> None:
        """Set the model for training, validation and inference."""
        self.structure_module.data_name = self._data_name()
        if stage == "predict" and not (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8
        ):
            self.use_kernels = False

    def forward(
        self, 
        feats: dict[str, Tensor],
        # Training arguments
        multiplicity_flow_train: int = 1,
        # Sampling arguments
        multiplicity_flow_sample: int = 1,
        num_sampling_steps: Optional[int] = None,
        use_sde_coords: bool = False,
        center_during_sampling: bool = True,
        refine_final: bool = False,
        return_trajectory: bool = False,
        steering_args: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        
        dict_out = {}

        ##### Structure prediction #####
        if self.training:
            # Returns denoised predictions for loss computation
            dict_out.update(
                self.structure_module(
                    feats=feats,
                    multiplicity=multiplicity_flow_train
                )
            )
        else:
            # Returns sampled structure
            atom_mask = feats["atom_pad_mask"]

            network_condition_kwargs = dict(
                feats=feats,
            )

            dict_out.update(
                self.structure_module.sample(
                    atom_mask=atom_mask,
                    num_sampling_steps=num_sampling_steps,
                    multiplicity=multiplicity_flow_sample,
                    use_sde_coords=use_sde_coords,
                    center_coords=center_during_sampling,
                    refine_final=refine_final,
                    return_trajectory=return_trajectory,
                    steering_args=steering_args,
                    **network_condition_kwargs,
                )
            )

        return dict_out
    
    def _compute_and_log_loss(self, batch: dict[str, Tensor], prefix: str) -> tuple[Tensor, dict, dict]:
        # Compute the forward pass
        out = self.structure_module(
            feats=batch,
            multiplicity=self.training_args['train_multiplicity'],
        )

        # Compute losses
        flow_loss_dict = self.structure_module.compute_loss(
            batch,
            out,
            multiplicity=self.training_args['train_multiplicity'],
            loss_type=self.training_args['loss_type'],
            use_dist_loss=self.training_args['use_dist_loss'],
            dist_cutoff=self.training_args['dist_cutoff'],
            use_dist_schedule=self.training_args['use_dist_schedule']
        )

        # Calculate total weighted loss
        total_loss = (
            self.training_args['coord_loss_weight'] * flow_loss_dict['coord_loss'].mean()
            + self.training_args['lengths_loss_weight'] * flow_loss_dict['length_loss'].mean()
            + self.training_args['angle_loss_weight'] * flow_loss_dict['angle_loss'].mean()
            + self.training_args['dist_loss_weight'] * flow_loss_dict['dist_loss'].mean()
        )

        # Loggings
        batch_size = batch['ref_element'].shape[0]
        sync_dist = prefix == 'val'

        # Loss per component
        for loss_name, batch_loss in flow_loss_dict.items():
            self.log(f'{prefix}/{loss_name}', batch_loss.mean(), batch_size=batch_size, sync_dist=sync_dist)

        # Total loss
        self.log(f'{prefix}/total_loss', total_loss, batch_size=batch_size, sync_dist=sync_dist)

        return total_loss, flow_loss_dict, out

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        step_start_time = time.time() 

        total_loss, flow_loss_dict, out = self._compute_and_log_loss(batch, prefix='train')

        # Training-specific loggings
        batch_size = batch['ref_element'].shape[0]
        self.log('train/batch_size', batch_size)

        # Stratified loss logging
        for loss_name, batch_loss in flow_loss_dict.items():
            # Stratified loss
            stratified_losses = stratify_loss_by_time(
                batch_t=out['times'], batch_loss=batch_loss, loss_name=loss_name
            )
            for k, v in stratified_losses.items():
                self.log(f'train/{k}', v, batch_size=batch_size)

        # Time
        step_time = time.time() - step_start_time
        self.log('train/samples_per_second', batch_size / step_time)

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        # Compute and log validation loss
        self._compute_and_log_loss(batch, prefix='val')

        # Sampling and structure matching
        sample_every_n_epochs = self.validation_args.get('sample_every_n_epochs', 1)
        if sample_every_n_epochs <= 0 or (self.current_epoch + 1) % sample_every_n_epochs != 0:
            return
        
        n_samples = self.validation_args['flow_samples']

        try:
            out = self(
                batch,
                num_sampling_steps=self.validation_args['sampling_steps'],
                center_during_sampling=self.validation_args['sampling_center_coords'],
                multiplicity_flow_sample=n_samples,
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                rank_zero_info("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e
        
        # Aggregate outputs
        return_dict = {
            "sampled_coords": out['sampled_atom_coords'],
            "sampled_lattices": out['sampled_lattice'],
            "true_coords": batch['cart_coords'],
            "true_lattices": batch['lattice'],
            "atom_types": batch['ref_element'],
            "atom_mask": batch['atom_pad_mask'],
            "block_type": batch['block_type'],
        }
        self.validation_step_outputs.append(return_dict)

    def on_validation_epoch_end(self) -> None:
        # This hook only runs on rank 0 when using DDP
        # The 'outputs' argument is a list of dictionaries from every validation_step
        if self.global_rank == 0:
            # Filter out None
            valid_outputs = [out for out in self.validation_step_outputs if out is not None]
            if not valid_outputs:
                return

            n_samples = self.validation_args['flow_samples']
            match_mode = self.validation_args.get('match_mode', 'all')
            tasks = []

            # Collate results from all batches
            for batch_output in valid_outputs:
                # Reshape and move to cpu
                sampled_coords = rearrange(batch_output['sampled_coords'], '(b m) n c -> b m n c', m=n_samples).cpu().numpy()
                sampled_lattices = rearrange(batch_output['sampled_lattices'], '(b m) c -> b m c', m=n_samples).cpu().numpy()
                
                true_coords = batch_output['true_coords'].cpu().numpy()
                true_lattices = batch_output['true_lattices'].cpu().numpy()
                atom_types = batch_output['atom_types'].cpu().numpy()
                atom_mask = batch_output['atom_mask'].cpu().numpy().astype(bool)
                block_type = batch_output['block_type'].cpu().numpy()

                # Prepare tasks for each item in the batch
                batch_size = sampled_coords.shape[0]
                for i in range(batch_size):
                    tasks.append(
                        (
                            sampled_coords[i],
                            sampled_lattices[i],
                            true_coords[i],
                            true_lattices[i],
                            atom_types[i],
                            atom_mask[i],
                            block_type[i],
                            match_mode,
                            self.matcher_kwargs,
                            self._data_name(),
                        )
                    )

            # Settings for parallel processing
            timeout_seconds = self.validation_args['timeout']
            num_workers = self.validation_args['num_workers']

            # Execute in parallel
            results_per_item = []
            with ProcessPool(max_workers=num_workers, context=mp.get_context("spawn")) as pool:
                future_map = {pool.schedule(find_best_match_rmsd, args=(task,)): task for task in tasks}

                for future in future_map:
                    try:
                        result = future.result(timeout=timeout_seconds)
                        results_per_item.append(result)
                    except PebbleTimeoutError:
                        future.cancel()
                        rank_zero_info(f"| WARNING: A structure matching task timed out. Skipping.")
                        results_per_item.append([None] * n_samples)
                    except Exception as e:
                        future.cancel()
                        rank_zero_info(f"| WARNING: A task failed with an exception: {e}")
                        results_per_item.append([None] * n_samples)
            
            # Compute match rate and RMSD
            for result_list in results_per_item:
                valid_rmsds = [r for r in result_list if r is not None]
                if valid_rmsds:
                    self.val_match_rate.update(1.0)
                    self.val_rmsd.update(min(valid_rmsds))
                else:
                    self.val_match_rate.update(0.0)
            
            # Log aggregated metrics
            self.log('val/match_rate', self.val_match_rate.compute(), rank_zero_only=True)
            if self.val_rmsd.update_count > 0:
                self.log('val/avg_rmsd', self.val_rmsd.compute(), rank_zero_only=True)
            else:
                self.log('val/avg_rmsd', float('nan'), rank_zero_only=True)

            # Reset metrics for next epoch
            self.val_match_rate.reset()
            self.val_rmsd.reset()
            self.validation_step_outputs.clear() # free memory

    def _steering_args(self) -> Optional[dict[str, Any]]:
        steering_args = self.predict_args.get('steering_args', None)
        if steering_args is None:
            return None
        return {**steering_args, "data": self._data_name()}

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        num_samples = self.predict_args.get('num_samples', 1)

        try:
            out = self(
                batch,
                num_sampling_steps=self.predict_args['sampling_steps'],
                use_sde_coords=self.predict_args['use_sde_coords'],
                center_during_sampling=self.predict_args['sampling_center_coords'],
                refine_final=self.predict_args['refine_final'],
                steering_args=self._steering_args(),
                multiplicity_flow_sample=num_samples,
            )
            
            prediction_output = {
                "exception": False,
                "generated_coords": out['sampled_atom_coords'],
                "generated_lattices": out['sampled_lattice'],
                "atom_types": batch['ref_element'],
                "atom_mask": batch['atom_pad_mask'],
                "atom_block_type": batch['block_type'],
            }

            if 'final_energy' in out:
                prediction_output['final_energy'] = out['final_energy']
                prediction_output['num_particles'] = self.predict_args['steering_args']['num_particles']

            if 'cart_coords' in batch:
                prediction_output['true_coords'] = batch['cart_coords']
            
            if 'lattice' in batch:
                prediction_output['true_lattices'] = batch['lattice']

            return prediction_output

        except RuntimeError as e:
            if "out of memory" in str(e):
                rank_zero_info(f"| WARNING: Ran out of memory during prediction on batch {batch_idx}. Skipping batch.")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise e
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.trainer.model.parameters(), 
            lr=self.training_args.lr,
            weight_decay=self.training_args.weight_decay,
        )

        if self.training_args.lr_scheduler == "af3":
            scheduler = AlphaFoldLRScheduler(
                optimizer=optimizer,
                base_lr=self.training_args.base_lr,
                max_lr=self.training_args.max_lr,
                warmup_no_steps=self.training_args.lr_warmup_no_steps,
                start_decay_after_n_steps=self.training_args.lr_start_decay_after_n_steps,
                decay_every_n_steps=self.training_args.lr_decay_every_n_steps,
                decay_factor=self.training_args.lr_decay_factor,
            )
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

        return {"optimizer": optimizer}

    def configure_callbacks(self) -> list[Callback]:
        """Configure model callbacks.

        Returns
        -------
        List[Callback]
            List of callbacks to be used in the model.

        """
        return [EMA(self.ema_decay)] if self.use_ema else []
