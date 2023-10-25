import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

import torch.multiprocessing as mp

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.datasets import load_darcy_flow_small
from neuralop.training import setup
from neuralop.training.callbacks import MGPatchingCallback, SimpleWandBLoggerCallback
from neuralop.utils import get_wandb_api_key, count_params
from neuralop.mpu.comm import cleanup


def main(rank=0):
    """
    main function to train and evaluate TFNO 
    on the small Darcy flow dataset. DDP is implemented. 
    """
    # Read the configuration
    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./darcy_config.yaml", config_name="default", config_folder="../config"
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder="../config"),
        ]
    )
    config = pipe.read_conf()
    config_name = pipe.steps[-1].config_name

    # Set-up distributed communication, if using
    is_logger = not config.distributed.use_distributed
    # Set up WandB logging
    wandb_init_args = {}
    if config.wandb.log and is_logger:
        wandb.login(key=get_wandb_api_key())
        if config.wandb.name:
            wandb_name = config.wandb.name
        else:
            wandb_name = "_".join(
                f"{var}"
                for var in [
                    config_name,
                    config.tfno2d.n_layers,
                    config.tfno2d.hidden_channels,
                    config.tfno2d.n_modes_width,
                    config.tfno2d.n_modes_height,
                    config.tfno2d.factorization,
                    config.tfno2d.rank,
                    config.patching.levels,
                    config.patching.padding,
                ]
            )
        wandb_init_args = dict(
            config=config,
            name=wandb_name,
            group=config.wandb.group,
            project=config.wandb.project,
            entity=config.wandb.entity,
        )
        if config.wandb.sweep:
            for key in wandb.config.keys():
                config.params[key] = wandb.config[key]
    # initialize process group
    device, is_logger = setup(config, world_rank=rank)

    # set up device

    if not config.distributed.use_distributed:
        if torch.backends.cuda.is_built():
            device = 'cuda:0'
        else:
            device = 'cpu'
    else:
        device = f'cuda:{rank}'
    
    # Print config to screen
    if config.verbose:
        pipe.log()
        sys.stdout.flush()

    # Loading the Darcy flow dataset
    train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=config.data.n_train,
        batch_size=config.data.batch_size,
        positional_encoding=config.data.positional_encoding,
        test_resolutions=config.data.test_resolutions,
        n_tests=config.data.n_tests,
        test_batch_sizes=config.data.test_batch_sizes,
        encode_input=config.data.encode_input,
        encode_output=config.data.encode_output,
    )
    
    model = get_model(config)
    model = model.to(device)

    # Use distributed data parallel
    if config.distributed.use_distributed:
        model = DDP(
            model, device_ids=[rank], output_device=rank, static_graph=True
        )

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.opt.learning_rate,
        weight_decay=config.opt.weight_decay,
    )

    if config.opt.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.opt.gamma,
            patience=config.opt.scheduler_patience,
            mode="min",
        )
    elif config.opt.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt.scheduler_T_max
        )
    elif config.opt.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
        )
    else:
        raise ValueError(f"Got scheduler={config.opt.scheduler}")


    # Creating the losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    if config.opt.training_loss == "l2":
        train_loss = l2loss
    elif config.opt.training_loss == "h1":
        train_loss = h1loss
    else:
        raise ValueError(
            f'Got training_loss={config.opt.training_loss} '
            f'but expected one of ["l2", "h1"]'
        )
    eval_losses = {"h1": h1loss, "l2": l2loss}

    if config.verbose and is_logger:
        print("\n### MODEL ###\n", model)
        print("\n### OPTIMIZER ###\n", optimizer)
        print("\n### SCHEDULER ###\n", scheduler)
        print("\n### LOSSES ###")
        print(f"\n * Train: {train_loss}")
        print(f"\n * Test: {eval_losses}")
        print(f"\n### Beginning Training...\n")
        sys.stdout.flush()

    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        device=device,
        amp_autocast=config.opt.amp_autocast,
        wandb_log=config.wandb.log,
        log_test_interval=config.wandb.log_test_interval,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose and is_logger,
        callbacks=[
            MGPatchingCallback(levels=config.patching.levels,
                                    padding_fraction=config.patching.padding,
                                    stitching=config.patching.stitching,
                                    encoder=output_encoder),
            SimpleWandBLoggerCallback(is_logger=is_logger,
                                      **wandb_init_args)
                ]
                )

    # Log parameter count after initalizing wandb in Callback
    if is_logger:
        n_params = count_params(model)

        if config.verbose:
            print(f"\nn_params: {n_params}")
            sys.stdout.flush()

        if config.wandb.log:
            to_log = {"n_params": n_params}
            if config.n_params_baseline is not None:
                to_log["n_params_baseline"] = (config.n_params_baseline,)
                to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
                to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
            wandb.log(to_log)
            wandb.watch(model)



    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    if config.wandb.log and is_logger:
        wandb.finish()

    cleanup()

if __name__ == "__main__":
    # Read the configuration
    config_name = "default"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./darcy_config.yaml", config_name="default", config_folder="../config"
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder="../config"),
        ]
    )
    config = pipe.read_conf()

    config_name = pipe.steps[-1].config_name

    if config.distributed.use_distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(main, 
                 nprocs=world_size)
    else:
        main()
