import torch
from torch.autograd import grad
import pytest
from tensorly import tenalg

tenalg.set_backend("einsum")

from ..gno_block import GNOBlock

# Fixed variables
in_channels = 3
out_channels = 3
mlp_hidden_layers = [16,16,16]

# data parameters
n_in = 100
n_out = 50

# test open3d mode if built
try:
    from neighbor_search import FixedRadiusSearch
    open3d_built = True
except:
    open3d_built = False

if open3d_built:
    use_open3d_parametrize = [True, False]
else:
    use_open3d_parametrize = [False]

# Test torch_scatter util if built
try:
    from torch_scatter import segment_csr
    torch_scatter_built = True
except:
    torch_scatter_built = False

if torch_scatter_built:
    use_torch_scatter_parametrize = [True, False]
else:
    use_torch_scatter_parametrize = [False]


@pytest.mark.parametrize("batch_size", [1,4])
@pytest.mark.parametrize("gno_coord_dim", [2,3])
@pytest.mark.parametrize("gno_pos_embed_type", ['nerf', 'transformer', None])
@pytest.mark.parametrize(
    "gno_transform_type", ["linear", "nonlinear_kernelonly", "nonlinear"]
)
@pytest.mark.parametrize('use_open3d', use_open3d_parametrize)
@pytest.mark.parametrize('use_torch_scatter', use_torch_scatter_parametrize)
def test_gno_block(gno_transform_type, gno_coord_dim, gno_pos_embed_type, batch_size, use_open3d, use_torch_scatter):
    if torch.backends.cuda.is_built():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu:0")
    
    use_open3d = use_open3d and (gno_coord_dim == 3)
    

    gno_block = GNOBlock(
        in_channels=in_channels,
        out_channels=out_channels, # dummy var currently
        coord_dim=gno_coord_dim,
        pos_embedding_type=gno_pos_embed_type,
        radius=0.25,
        channel_mlp_layers=mlp_hidden_layers,
        transform_type=gno_transform_type,
        use_open3d_neighbor_search=use_open3d,
        use_torch_scatter_reduce=use_torch_scatter
    ).to(device)

    # create input geometry and output queries
    input_geom_shape = [n_in, gno_coord_dim]
    input_geom = torch.randn(*input_geom_shape, device=device)

    output_queries_shape = [n_out, gno_coord_dim]
    output_queries = torch.randn(*output_queries_shape, device=device)

    f_y = None
    if gno_transform_type != "linear":
    # create data and features
        f_y_shape = [batch_size, n_in, in_channels]
        f_y = torch.randn(*f_y_shape, device=device)
        # require and retain grad to check for backprop
        f_y.requires_grad_(True)


    out = gno_block(y=input_geom,
                    x=output_queries,
                    f_y=f_y)
    
    # Check output size
    # Batched outputs only matter in the nonlinear kernel use case
    if gno_transform_type != "linear":
        assert list(out.shape) == [batch_size, n_out, out_channels]
    else:
        assert list(out.shape) == [n_out, out_channels]
    
    # Check backward pass
    assert out.isfinite().all()
    if batch_size > 1:
        loss = out[0].sum()
    else:
        loss = out.sum()
    loss.backward()
    
    if batch_size > 1 and gno_transform_type != "linear":
        # assert f_y[1:] accumulates no grad if it's used
        assert not f_y.grad[1:].nonzero().any()
