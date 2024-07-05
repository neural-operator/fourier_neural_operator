import torch
from torch.cuda import amp
from timeit import default_timer
import pathlib
import sys
import wandb

from .callbacks import PipelineCallback
import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
from neuralop.losses.meta_losses import SumLossOutput


class Trainer:
    def __init__(
        self,
        *,
        model,
        n_epochs,
        wandb_log=True,
        device=None,
        amp_autocast=False,
        data_processor=None,
        callbacks=None,
        log_test_interval=1,
        log_output=False,
        use_distributed=False,
        verbose=False,
    ):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        device : torch.device
        amp_autocast : bool, default is False
        data_processor : class to transform data, default is None
            if not None, data from the loaders is transform first with data_processor.preprocess,
            then after getting an output from the model, that is transformed with data_processor.postprocess.
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
        """

        if callbacks:
            assert isinstance(
                callbacks, list
            ), "Callbacks must be a list of Callback objects"
            self.callbacks = PipelineCallback(callbacks=callbacks)
            self.override_load_to_device = (
                self.callbacks.device_load_callback_idx is not None
            )
            self.overrides_train_loss = self.callbacks.overrides_train_loss
        else:
            self.callbacks = []
            self.override_load_to_device = False
            self.overrides_train_loss = False

        if verbose:
            print(f"{self.override_load_to_device=}")
            print(f"{self.overrides_train_loss=}")

        if self.callbacks:
            self.callbacks.on_init_start(
                model=model,
                n_epochs=n_epochs,
                wandb_log=wandb_log,
                device=device,
                amp_autocast=amp_autocast,
                log_test_interval=log_test_interval,
                log_output=log_output,
                use_distributed=use_distributed,
                verbose=verbose,
            )

        self.model = model
        self.n_epochs = n_epochs

        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.amp_autocast = amp_autocast
        self.data_processor = data_processor

        if self.callbacks:
            self.callbacks.on_init_end(
                model=model,
                n_epochs=n_epochs,
                wandb_log=wandb_log,
                device=device,
                amp_autocast=amp_autocast,
                log_test_interval=log_test_interval,
                log_output=log_output,
                use_distributed=use_distributed,
                verbose=verbose,
            )

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer,
        training_loss=None,
        eval_losses=None,
    ):
        """Trains the given model on the given datasets.
        params:
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        """

        if self.callbacks:
            self.callbacks.on_train_start(
                model=self.model,
                train_loader=train_loader,
                test_loaders=test_loaders,
                optimizer=optimizer,
                scheduler=scheduler,
                regularizer=regularizer,
                training_loss=training_loss,
                eval_losses=eval_losses,
                data_processor=self.data_processor,
            )

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        errors = None

        if self.verbose:
            print(f'Training on {len(train_loader)} samples')
            print(f'Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples'
                  f'         on resolutions {[name for name in test_loaders]}.')
            sys.stdout.flush()

        for epoch in range(self.n_epochs):
            if self.callbacks:
                self.callbacks.on_epoch_start(epoch=epoch)

            avg_loss = 0
            avg_lasso_loss = 0
            self.model.train()
            if self.data_processor:
                self.data_processor.train()
            t1 = default_timer()
            train_err = 0.0
            
            # track number of training examples in batch
            n_samples = 0

            for idx, sample in enumerate(train_loader):
                if self.callbacks:
                    self.callbacks.on_batch_start(
                        epoch=epoch,
                        idx=idx,
                        sample=sample,
                        data_processor=self.data_processor
                    )

                optimizer.zero_grad(set_to_none=True)
                if regularizer:
                    regularizer.reset()

                if self.data_processor is not None:
                    sample = self.data_processor.preprocess(sample)
                else:
                    # load data to device if no preprocessor exists
                    sample = {
                        k: v.to(self.device)
                        for k, v in sample.items()
                        if torch.is_tensor(v)
                    }

                n_samples += sample["y"].shape[0]

                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        out = self.model(**sample)
                else:
                    out = self.model(**sample)

                if self.data_processor is not None:
                    out, sample = self.data_processor.postprocess(out, sample)

                '''if self.callbacks:
                    out = self.callbacks.on_before_loss(out=out)'''

                # log output shape the first time outputs are received
                if epoch == 0 and idx == 0 and self.verbose:
                    print(f"Raw outputs of shape {out.shape}")
                

                if self.overrides_train_loss:
                    loss = self.callbacks.compute_training_loss(
                        out=out, 
                        sample=sample, 
                        loss=training_loss,
                        model=self.model,
                        amp_autocast=self.amp_autocast
                    )
                else:
                    if self.amp_autocast:
                        with amp.autocast(enabled=True):
                            loss = training_loss(out, **sample)
                    else:
                        loss = training_loss(out, **sample)
                if regularizer:
                    loss += regularizer.loss
                
                loss.backward()

                del out

                # free sample memory
                sample = {k: None for k in sample.keys()}
                del sample

                optimizer.step()
                if isinstance(loss, torch.Tensor):
                    train_err += loss.item()
                elif isinstance(loss, SumLossOutput):
                    train_err = loss + train_err

                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1

            train_err /= len(train_loader)
            avg_loss /= n_samples
            if regularizer:
                avg_lasso_loss /= n_samples
            
            # collect info to log, message to print
            if epoch % self.log_test_interval == 0:
                msg = f"[{epoch}] time={epoch_train_time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}"
                if regularizer: 
                    msg += f", avg_lasso={avg_lasso_loss:.5f}"
                values_to_log = dict(train_err=train_err,
                                     time=epoch_train_time,
                                     avg_loss=avg_loss,
                                     avg_lasso_loss=avg_lasso_loss)
                #TODO: rename and/or reform
                #msg, values_to_log = self.callbacks.logging_method_name(msg, values_to_log)
            
                for loader_name, loader in test_loaders.items():
                    errors = self.evaluate(eval_losses, loader, log_prefix=loader_name)
                    for loss_name, loss_value in errors.items():
                        if isinstance(loss_value, float):
                            msg += f", {loss_name}={loss_value:.4f}"
                        else:
                            # TODO: figure out how to change this form for Weighted Sum
                            loss_value = {i: e.item() for (i, e) in enumerate(loss_value)}
                            msg += f", val_{loss_name}={loss_value}"
                        values_to_log[f"val_{loss_name}"] = loss_value
                
                # print msg to console and optionally log to wandb
                if self.verbose:
                    print(msg)
                    sys.stdout.flush()
                
                if self.wandb_log:
                    for pg in optimizer.param_groups:
                        lr = pg["lr"]
                        values_to_log["lr"] = lr
                    wandb.log(
                        values_to_log,
                        step=epoch + 1,
                        commit=True,
                    )

            if self.callbacks:
                self.callbacks.on_epoch_end(
                    epoch=epoch, 
                    train_err=train_err, 
                    avg_loss=avg_loss,
                    errors=errors,
                    model=self.model
                )

        return errors

    def evaluate(self, loss_dict, data_loader, log_prefix=""):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """

        if self.callbacks:
            self.callbacks.on_val_epoch_start(
                log_prefix=log_prefix, loss_dict=loss_dict, data_loader=data_loader
            )

        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0. for loss_name in loss_dict.keys()}

        n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                '''if self.callbacks:
                    sample = self.callbacks.on_val_batch_start(
                        idx=idx, sample=sample, data_processor=self.data_processor
                    )'''

                if self.data_processor is not None:
                    sample = self.data_processor.preprocess(sample)
                else:
                    # load data to device if no preprocessor exists
                    sample = {
                        k: v.to(self.device)
                        for k, v in sample.items()
                        if torch.is_tensor(v)
                    }
                n_samples += sample["y"].size(0)

                out = self.model(**sample)

                if self.data_processor is not None:
                    out, sample = self.data_processor.postprocess(out, sample)

                if self.callbacks:
                    self.callbacks.on_before_val_loss(out=out)

                for loss_name, loss in loss_dict.items():
                    val_loss = loss(out, **sample)
                    if val_loss.shape == ():
                        val_loss = val_loss.item()

                    errors[f"{log_prefix}_{loss_name}"] += val_loss

                if self.callbacks:
                    self.callbacks.on_val_batch_end()

        for key in errors.keys():
            errors[key] /= n_samples

        if self.callbacks:
            self.callbacks.on_val_epoch_end(errors=errors, sample=sample, out=out)

        del out

        return errors
