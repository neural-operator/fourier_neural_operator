from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings

import torch
from torch.cuda import amp
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
from .training_state import load_training_state, save_training_state
from .trainer import Trainer


class AutoregressiveTrainer(Trainer):
    """
    A general Trainer class to train neural-operators on given datasets.

    Parameters
    ----------
    model : nn.Module
    n_epochs : int
    wandb_log : bool, default is False
        whether to log results to wandb
    device : torch.device, or str 'cpu' or 'cuda'
    mixed_precision : bool, default is False
        whether to use torch.autocast to compute mixed precision
    data_processor : DataProcessor class to transform data, default is None
        if not None, data from the loaders is transform first with data_processor.preprocess,
        then after getting an output from the model, that is transformed with data_processor.postprocess.
    eval_interval : int, default is 1
        how frequently to evaluate model and log training stats
    log_output : bool, default is False
        if True, and if wandb_log is also True, log output images to wandb
    use_distributed : bool, default is False
        whether to use DDP
    verbose : bool, default is False
    """
    def __init__(
        self,
        *, 
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool=False,
        device: str='cpu',
        mixed_precision: bool=False,
        data_processor: nn.Module=None,
        eval_interval: int=1,
        log_output: bool=False, # this currently doesn't work!
        use_distributed: bool=False,
        verbose: bool=False,
        T: int=10, 
        timestep: int=1,
        debug:bool=True
    ):
        self.T = T
        self.timestep = timestep
        self.debug = debug

        super().__init__(model=model,
                         n_epochs=n_epochs,
                         wandb_log=wandb_log,
                         device=device,
                         mixed_precision=mixed_precision,
                         data_processor=data_processor,
                         eval_interval=eval_interval,
                         log_output=log_output,
                         use_distributed=use_distributed,
                         verbose=verbose)
    
    def train_one_batch(self, idx, sample, training_loss):
        """Conduct autoregressive rollout for one batch

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()

        u = sample["u"]
        if self.debug:
            print(f"one {u.shape=}")
        for t in range(0, self.T, self.timestep):
            ## TODO@DAVID: should we have a general handler for autoregressive trainer?
            if self.data_processor is not None:
                sample = self.data_processor.preprocess(sample, step=t)
            else:
                # load data to device if no preprocessor exists
                sample = {
                    k: v.to(self.device)
                    for k, v in sample.items()
                    if torch.is_tensor(v)
                }
            self.n_samples += sample["y"].shape[0] # TODO@David: divide by timesteps

            if self.debug:
                print("one x shape: " + str(sample["x"].shape))
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    out = self.model(**sample)
            else:
                out = self.model(**sample)
            
            if self.epoch == 0 and idx == 0 and t == 0 and self.verbose:
                print(f"Raw outputs of shape {out.shape}")

            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample, step=t)

            loss = 0.0

            if self.debug:
                print(f"{out.shape=}")
                print(f"y: " + str(sample["y"].shape))
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    loss += training_loss(out, **sample)
            else:
                loss += training_loss(out, **sample)

            if self.regularizer:
                loss += self.regularizer.loss
            
            # roll x forward in time using the output of out

            '''if t == 0:
                full_out = out
            else:
                full_out = torch.cat((full_out, out), dim=-1)'''
        
        return loss
    
    def eval_one_batch(self,
                       sample: dict,
                       eval_losses: dict,
                       return_output: bool=False):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        eval_step_losses = {loss_name: 0. for loss_name in eval_losses.keys()}
        eval_rollout_losses = {loss_name: 0. for loss_name in eval_losses.keys()}

        u = sample["u"]
        if self.debug:
            print(f"one {u.shape=}")
        for t in range(0, self.T, self.timestep):
            ## TODO@DAVID: should we have a general handler for autoregressive trainer?
            if self.data_processor is not None:
                sample = self.data_processor.preprocess(sample, step=t)
            else:
                # load data to device if no preprocessor exists
                sample = {
                    k: v.to(self.device)
                    for k, v in sample.items()
                    if torch.is_tensor(v)
                }
            self.n_samples += sample["y"].shape[0]

            out = self.model(**sample)
            
            if self.debug:
                x = sample["x"]
                print(f"eval {x.shape=} {out.shape=}")
            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample, step=t)
            
            if t == 0:
                full_out = out
            else:
                full_out = torch.cat((full_out, out), dim=-1)

            for loss_name, loss in eval_losses.items():
                step_loss = loss(out, **sample)
                eval_step_losses[loss_name] += step_loss

                full_y = sample['full_y']
                #TODO @ DAVID: figure out better way to do this
                full_loss = loss(full_out, full_y)
                eval_rollout_losses[loss_name] += full_loss
            
            # roll x forward in time using the output of out
            #x = torch.cat((x[..., self.timestep:], out), dim=-1)
        # TODO@DAVID: return both step and full losses. You need to handle this in .evaluate() indexing
        all_eval_losses = {}
        for loss_name in eval_losses.keys():
            all_eval_losses[f'step_{loss_name}'] = eval_step_losses[loss_name]
            all_eval_losses[f'full_{loss_name}'] = eval_rollout_losses[loss_name]
        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None
    