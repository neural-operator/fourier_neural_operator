import math
import torch
from torch.testing import assert_close

from neuralop.data.datasets.darcy import load_darcy_flow_small
from ..equation_losses import BurgersEqnLoss, DarcyEqnLoss
from neuralop.layers.embeddings import regular_grid_nd

if torch.backends.cuda.is_built():
    device = "cuda"
else:
    device = "cpu"

def test_darcy_fdm():
    eqn_loss = DarcyEqnLoss(method="finite_difference")
    # load a few small darcy examples
    _, test_loaders, _ = load_darcy_flow_small(
        n_train=16,
        batch_size=4,
        test_resolutions=[32],
        n_tests=[16],
        test_batch_sizes=[4],
    )
    for i, batch in enumerate(test_loaders[32]):
        a = batch["x"].to(device)
        u = batch["y"].to(device)
        loss = eqn_loss(u, a, None)
        