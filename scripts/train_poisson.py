import sys
import torch
import wandb
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from neuralop.losses.data_losses import LpLoss, MSELoss
from neuralop.training import Trainer, setup
from neuralop.data.datasets.nonlinear_poisson import load_nonlinear_poisson_pt
from neuralop.losses.equation_losses import PoissonBoundaryLoss, PoissonEqnLoss
from neuralop.losses.meta_losses import WeightedSumLoss
from neuralop.models import get_model
from neuralop.utils import get_wandb_api_key, count_model_params


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./poisson_gino_config.yaml", config_name="default", config_folder="../config"
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
wandb_args = None
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                config_name,
                config.fno.n_layers,
                config.fno.hidden_channels,
                config.fno.n_modes_width,
                config.fno.n_modes[0],
                config.fno.factorization,
                config.fno.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_args =  dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_args)

else: 
    wandb_init_args = None
# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose:
    pipe.log()
    sys.stdout.flush()

# Load the Nonlinear Poisson dataset
train_loader, test_loader, data_processor = load_nonlinear_poisson_pt(
    data_path=config.data.file,
    query_res=config.data.query_resolution,
    n_train=config.data.n_train, 
    n_test=config.data.n_test,
    n_in=config.data.n_in,
    n_out=config.data.n_out,
    n_eval=config.data.n_eval,
    val_on_same_instance=config.data.single_instance,
    train_out_res=config.data.train_out_res,
    input_min_sample_points=config.data.input_min,
    input_max_sample_points=config.data.input_max,
    input_subsample_level=config.data.sample_random_in,
    output_subsample_level=config.data.sample_random_out
)
test_loaders = {"test": test_loader} ##TODO FIX ONCE NOT FITTING SINGLE INSTANCE

model = get_model(config)
model = model.to(device)

# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
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
# with default measure, 2D l2 is 2D MSE
mse_loss = MSELoss()

class GINOLoss(object):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss
    def __call__(self, out, y, **kwargs):
        loss = 0.
        if isinstance(out, dict) and isinstance(y, dict):
            for field, points in out.items():
                loss += self.base_loss(points, y[field], **kwargs)
            return loss
        else:
            return self.base_loss(out, y, **kwargs)

gino_mseloss = GINOLoss(mse_loss)

training_loss = config.opt.training_loss
if not isinstance(training_loss, (tuple, list)):
    training_loss = [training_loss]

losses = []
weights = []

if 'mse' in training_loss:
    #losses.append(mse_loss)
    losses.append(gino_mseloss)
    weights.append(config.opt.loss_weights.get('mse', 1.))
if 'equation' in training_loss:
    equation_loss = PoissonEqnLoss(interior_weight=config.opt.loss_weights.get('interior', 1.), 
                                    boundary_weight=config.opt.loss_weights.get('boundary', 1.),
                                    diff_method=config.opt.get('pino_method', 'autograd'))
    losses.append(equation_loss)
    weights.append(1)

train_loss = WeightedSumLoss(losses=losses, weights=weights)
#train_loss = WeightedSumLoss(losses=losses, weights=weights, return_individual=True, compute_grads=True)
eval_losses = {"mse": mse_loss}

if config.verbose:
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
    data_processor=data_processor,
    device=device,
    mixed_precision=config.opt.amp_autocast,
    eval_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log=config.wandb.log
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
        wandb.log(to_log)
        wandb.watch(model)

# This breaks if we run it, since trainer needs to run train at least once first
# We only use this as a check that the loss is high before training. 
# trainer.evaluate(eval_losses, test_loaders['test'])

trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=None,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if config.wandb.log and is_logger:
    wandb.finish()