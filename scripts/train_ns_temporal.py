import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import wandb


from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.datasets.temporal import TemporalDataset
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.mpu.comm import get_local_rank
from neuralop.training import setup, AdamW
from neuralop.training.autoregressive_trainer import AutoregressiveTrainer


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./navier_stokes_temporal_config.yaml", config_name="default", config_folder="../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)
# Set up WandB logging
wandb_init_args = None
if config.wandb.log and is_logger:
    print(config.wandb.log)
    print(config)
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.fno.n_layers,
                config.fno.n_modes,
                config.fno.hidden_channels,
                config.fno.factorization,
                config.fno.rank,
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
    wandb.init(**wandb_init_args)

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

# Load the dataset
dataset = TemporalDataset(root_dir="/home/dave/data/navier_stokes/temporal", 
                          dataset_name="navier_stokes", 
                          temporal_resolution=config.data.temporal_resolution,
                          T=config.data.T, timestep=config.data.timestep,
                          n_train=config.data.n_train, 
                          n_tests=config.data.n_tests,
                          batch_size=config.data.batch_size, 
                          test_batch_sizes=config.data.test_batch_sizes, 
                          train_resolution=config.data.train_resolution,
                          test_resolutions=config.data.test_resolutions,)
train_loader = DataLoader(dataset.train_db, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset.test_dbs[64], batch_size=1, shuffle=True)

# reconfigure data channels for temporal
config[config.arch].data_channels = config[config.arch].data_channels * config.data.T

model = get_model(config)
model = model.to(device)

data_processor = dataset.data_processor
data_processor.debug = config.debug
data_processor = data_processor.to(device)

# Create the optimizer
optimizer = AdamW(
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

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()


trainer = AutoregressiveTrainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    mixed_precision=config.opt.amp_autocast,
    eval_interval=config.wandb.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log = config.wandb.log,
    T=10,
    timestep=1,
    debug=config.debug
)

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        wandb.watch(model)


trainer.train(
    train_loader,
    test_loaders={'val': val_loader},
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if config.wandb.log and is_logger:
    wandb.finish()
