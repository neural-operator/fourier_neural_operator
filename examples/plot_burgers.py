"""
Training a 1D FNO on the Burgers Equation
=========================================

This notebooks walks through the Fourier Neural Operator for a 1D problem such
as the (time-independent) Burgers equation discussed in Section 5.1 in the paper
[Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/pdf/2010.08895.pdf).
The solution operator maps the field equation $A(x)$ at time $t=0$
to $U(x)$ at time $t=1$.
"""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer
import yaml

from torch.nn.functional import mse_loss
import torch

from neuralop import count_params
from neuralop.datasets import load_burgers
from neuralop.layers import SpectralConv1d
from neuralop.models import FNO1d
from neuralop.training import LpLoss

np.random.seed(0)
torch.manual_seed(0)

# %%
# Load configurations that control the training and evaluation of the FNO model:

class Config(NamedTuple):
  data_path: str
  """Fully qualified file path to training/testing data"""
  n_train: int
  n_test: int
  train_batch_size: int
  test_batch_size: int
  learning_rate: float
  epochs: int
  iterations: int
  modes: int
  width: int
  subsampling_rate: int
  s: int
  h: int

  @staticmethod
  def from_yaml(config_path: str):
    with open(config_path, 'r') as f:
      cfg = yaml.load(f)

    config = Config(
      data_path=cfg['data_path'],
      n_train=cfg['n_train'],
      n_test=cfg['n_test'],
      train_batch_size=cfg['train_batch_size'],
      test_batch_size=cfg['test_batch_size'],
      learning_rate=cfg['learning_rate'],
      epochs=cfg['epochs'],
      iterations=cfg['iterations'],
      modes=cfg['modes'],
      width=cfg['width'],
      subsampling_rate=cfg['subsampling_rate'],
      s=cfg['s'],
      h=cfg['h'],
    )
    return config


config = Config.from_yaml('burgers.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}')

# %%
# Read Burgers data from disk. If the data is not found in the given directory,
# TODO download it from Zenodo and proceed to load, train, and evaluate.

# Data is of the shape (number of samples, grid size)
train_loader, test_loader = load_burgers(
    config.data_path,
    config.n_train,
    config.n_test,
    batch_test=1
)

model = FNO1d(
  # modes_height : number of Fourier modes to keep along the height
  int(config.modes),

  # hidden_channels : width of the FNO (i.e. number of channels)
  int(config.width),

  in_channels=2,
  use_mlp=True,
  SpectralConv=SpectralConv1d,
  decompostion_kwargs={'dtype', torch.cdouble},
  fno_block_precision='double',
  dtype=torch.float64,
).cuda()
print(F"Model parameter count: {count_params(model):,d}")

# %%
# Training and Evaluation.
# We use visually evaluate (i.e. print to console the loss of) the model using
# mean squared error (MSE) in addition to training on L2 loss. The MSE will be
# measured point-wise on predicted field function $U(x, t=1)$ for each
# discretized point (i.e. by x-value).

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.iterations)

loss = LpLoss()
header_strings = (
  'Epoch',
  'Duration',
  'Training MSE',
  'Training L2',
  'Testing L2',
)
HEADER = ' | '.join(header_strings)
for ep in range(int(config.epochs)):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        mse = mse_loss(
            out.view(config.train_batch_size, -1),
            y.view(config.train_batch_size, -1),
            reduction='mean'
        )
        l2 = loss(
            out.view(config.train_batch_size, -1),
            y.view(config.train_batch_size, -1)
        )
        l2.backward() # use the l2 relative loss

        optimizer.step()
        scheduler.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            test_l2 += loss(
                out.view(config.test_batch_size, -1),
                y.view(config.test_batch_size, -1)
            ).item()

    train_mse /= len(train_loader)
    train_l2 /= config.n_train
    test_l2 /= config.n_test
    t2 = default_timer()

    if ep % 10 == 0:
        print(HEADER)
    row = ' | '.join(
        [f'{ep:{len(header_strings[0])}d}', f'{t2 - t1:{len(header_strings[1])}.3f}'] +
        [f'{x:{len(s)}.6f}' for s, x
         in zip(header_strings[2:], (train_mse,
                                     train_l2,
                                     test_l2))])
    print(row)

# %%
# Plot Input/Truth/Predictions for a subset of test data.
# Note we can barely see the Ground Truth curve above, so the model's prediction
# already agrees with the solver that generated this Burgers dataset to within
# a small error.

test_samples = test_loader.dataset

fig, axs = plt.subplots(2, 2, figsize=(10, 6)) # , layout='constrained')
for index, ax in enumerate(axs.flat):
    data = test_samples[index * 10]
    # Input ``a`` & Ground-truth ``u``
    a, u = data[0].to(device), data[1].to(device)
    # Model prediction
    out = model(a.unsqueeze(0))
    ax.plot(
        a[0].cpu(),
        label="Input `x`")
    ax.plot(
        u.squeeze().cpu(),
        label="Ground-truth `y`")
    ax.plot(
        out.squeeze().detach().cpu().numpy(),
        label="Model prediction")
    plt.xticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
plt.legend(loc='lower right')
