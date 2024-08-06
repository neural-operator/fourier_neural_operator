import pytest
import torch
import math
from ..transformer_block import TransformerEncoderBlock, TransformerDecoderBlock
from ..embeddings import RotaryEmbedding2D

def test_TransformerEncoderBlock_output():
    """Test TransformerEncoderBlock
    """
    
    mlp_dropout = 0
    mlp_expansion = 0.5
    size = [10]*2

    for ndim in [1, 2]:   # currently only 1D and 2D are supported
        block = TransformerEncoderBlock(
            3, 4, 16,
            num_heads=4, head_n_channels=16, n_layers=3,
            use_mlp=True, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion)
        pos_emb_module = RotaryEmbedding2D(16 // ndim)
        flattened_size = math.prod(size[:ndim])
        x = torch.randn(2, flattened_size, 3)
        pos = []
        for i in range(ndim):
            pos.append(torch.linspace(0, 1, size[i]))
        pos = torch.meshgrid(*pos)
        pos = torch.cat([p.flatten().view(1, -1, 1) for p in pos], dim=-1)
        res = block(x, pos, pos_emb_module)
        assert res.shape[1] == flattened_size   # Check grid size
        assert res.shape[-1] == 4   # Check out channels


@pytest.mark.parametrize('norm', 
                         ['instance_norm', 'layer_norm', 'group_norm', 'none'])
def test_TransformerEncoderBlock_norm(norm):
    """Test TransformerEncoderBlock with different normalization layers
    """
    size = [10] * 2
    mlp_dropout = 0
    mlp_expansion = 0.5

    block = TransformerEncoderBlock(
        3, 4, 16,
        num_heads=4, head_n_channels=16, n_layers=3,
        use_mlp=True, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, norm=norm)
    pos_emb_module = RotaryEmbedding2D(16 // 2)
    flattened_size = math.prod(size)
    x = torch.randn(2, flattened_size, 3)
    pos = []
    for i in range(2):
        pos.append(torch.linspace(0, 1, size[i]))
    pos = torch.meshgrid(*pos)
    pos = torch.cat([p.flatten().view(1, -1, 1) for p in pos], dim=-1)
    res = block(x, pos, pos_emb_module)
    assert res.shape[1] == flattened_size  # Check grid size
    assert res.shape[-1] == 4  # Check out channels


@pytest.mark.parametrize('norm',
                         ['instance_norm', 'layer_norm', 'group_norm', 'none'])
def test_TransformerEncoderBlock_norm(norm):
    """Test TransformerEncoderBlock with different normalization layers
    """
    size = [10] * 2
    mlp_dropout = 0
    mlp_expansion = 0.5

    block = TransformerEncoderBlock(
        3, 4, 16,
        num_heads=4, head_n_channels=16, n_layers=3,
        use_mlp=True, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, norm=norm)
    pos_emb_module = RotaryEmbedding2D(16 // 2)
    flattened_size = math.prod(size)
    x = torch.randn(2, flattened_size, 3)
    pos = []
    for i in range(2):
        pos.append(torch.linspace(0, 1, size[i]))
    pos = torch.meshgrid(*pos)
    pos = torch.cat([p.flatten().view(1, -1, 1) for p in pos], dim=-1)
    res = block(x, pos, pos_emb_module)
    assert res.shape[1] == flattened_size  # Check grid size
    assert res.shape[-1] == 4  # Check out channels


@pytest.mark.parametrize('query_basis',
                         ['siren', 'fourier', 'linear'])
def test_TransformerDecoderBlock_basis(query_basis):
    """Test TransformerDecoderBlock with different query basis
    """
    size = [10] * 2
    mlp_dropout = 0
    mlp_expansion = 0.5

    block = TransformerDecoderBlock(
        2, 3, 4, 16,
        num_heads=4, head_n_channels=16,
        query_basis=query_basis,
        use_mlp=True, mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion)
    pos_emb_module = RotaryEmbedding2D(16 // 2)
    flattened_size = math.prod(size)
    x = torch.randn(2, flattened_size, 3)
    pos = []
    for i in range(2):
        pos.append(torch.linspace(0, 1, size[i]))
    pos = torch.meshgrid(*pos)
    pos = torch.cat([p.flatten().view(1, -1, 1) for p in pos], dim=-1)
    res = block(x, pos, pos_emb_module)
    assert res.shape[1] == flattened_size
    assert res.shape[-1] == 4  # Check out channels

