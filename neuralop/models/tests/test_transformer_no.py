import time
import torch
import pytest
import math

from ..transformer_no import TransformerNO


@pytest.mark.parametrize(
    "input_shape", [(32, 2048, 1), (32, 64, 64, 3)]   # a 1D case and a 2D case
)
def test_TransformerNO(input_shape):
    if torch.has_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu:0')

    if len(input_shape) == 3:
        n_dim = 1
    else:
        n_dim = 2
    data_channles = input_shape[-1]
    model = TransformerNO(n_dim=n_dim,
                          in_channels=data_channles,
                          out_channels=1,    # single channel output
                          encoder_hidden_channels=32,
                          decoder_hidden_channels=32,
                          encoder_num_heads=4,
                          decoder_num_heads=4,
                          encoder_head_n_channels=32,
                          decoder_head_n_channels=32,
                          encoder_n_layers=3,
                          query_basis='siren')
    model.to(device)

    in_data = torch.randn(input_shape).to(device)
    in_data = in_data.view(in_data.shape[0], -1, in_data.shape[-1])   # flatten the spatial dimensions
    pos = []
    for i in range(n_dim):
        pos.append(torch.linspace(0, 1, input_shape[i+1]))
    pos = torch.meshgrid(*pos, indexing='ij')
    pos = torch.cat([p.flatten().view(1, -1, 1) for p in pos], dim=-1).to(device)

    # Test forward pass
    out = model(in_data, pos)

    # Check output size
    output_size = math.prod(input_shape[1:-1])
    assert list(out.shape) == [input_shape[0], output_size, 1]

    # Check backward pass
    loss = out.sum()
    loss.backward()

    n_unused_params = 0
    for name, param in model.named_parameters():
        if param.grad is None:
            n_unused_params += 1
            print(name)
    assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"
