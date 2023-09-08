import torch
from neuralop import TFNO3d, TFNO2d, TFNO1d, TFNO
from neuralop.models import FNO
import pytest
from tensorly import tenalg
from math import prod
from configmypy import Bunch

tenalg.set_backend("einsum")


@pytest.mark.parametrize(
    "factorization", ["ComplexDense", "ComplexTucker", "ComplexCP", "ComplexTT"]
)
@pytest.mark.parametrize("implementation", ["factorized", "reconstructed"])
@pytest.mark.parametrize("n_dim", [1, 2, 3])
@pytest.mark.parametrize("fno_block_precision", ["full", "half", "mixed"])
@pytest.mark.parametrize("stabilizer", [None, "tanh"])
@pytest.mark.parametrize("lifting_channels", [None, 256])
def test_tfno(
    factorization,
    implementation,
    n_dim,
    fno_block_precision,
    stabilizer,
    lifting_channels,
):
    if torch.has_cuda:
        device = "cuda"
        s = 128
        modes = 16
        width = 64
        fc_channels = 256
        batch_size = 20
        use_mlp = True
        n_layers = 4
        mlp = Bunch(dict(expansion=0.5, dropout=0))
    else:
        device = "cpu"
        fno_block_precision = "full"
        s = 16
        modes = 5
        width = 15
        fc_channels = 32
        batch_size = 3
        n_layers = 2

        use_mlp = True
        mlp = Bunch(dict(expansion=0.5, dropout=0))

    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim
    model = TFNO(
        hidden_channels=width,
        n_modes=n_modes,
        factorization=factorization,
        implementation=implementation,
        rank=rank,
        fixed_rank_modes=False,
        joint_factorization=False,
        n_layers=n_layers,
        fno_block_precision=fno_block_precision,
        use_mlp=use_mlp,
        mlp=mlp,
        stabilizer=stabilizer,
        fc_channels=fc_channels,
        lifting_channels=lifting_channels,
    ).to(device)
    in_data = torch.randn(batch_size, 3, *size).to(device)

    # Test forward pass
    out = model(in_data)

    # Check output size
    assert list(out.shape) == [batch_size, 1, *size]

    # Check backward pass
    loss = out.sum()
    loss.backward()

    n_unused_params = 0
    for param in model.parameters():
        if param.grad is None:
            n_unused_params += 1
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"


@pytest.mark.parametrize(
    "output_scaling_factor", [[2, 1, 1], [1, 2, 1], [1, 1, 2], [1, 2, 2], [1, 0.5, 1]]
)
def test_fno_superresolution(output_scaling_factor):
    device = "cpu"
    s = 16
    modes = 5
    hidden_channels = 15
    fc_channels = 32
    batch_size = 3
    n_layers = 3
    use_mlp = True
    n_dim = 2
    rank = 0.2
    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    model = FNO(
        n_modes,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        factorization="cp",
        implementation="reconstructed",
        rank=rank,
        output_scaling_factor=output_scaling_factor,
        n_layers=n_layers,
        use_mlp=use_mlp,
        fc_channels=fc_channels,
    ).to(device)

    in_data = torch.randn(batch_size, 3, *size).to(device)
    # Test forward pass
    out = model(in_data)

    # Check output size
    factor = prod(output_scaling_factor)

    assert list(out.shape) == [batch_size, 1] + [int(round(factor * s)) for s in size]
