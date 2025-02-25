import torch
from ..segment_csr import segment_csr
import pytest


@pytest.mark.parametrize('batch_size', [1,4])
def test_native_segcsr_shapes(batch_size):
    n_pts = 25
    n_channels = 5
    max_nbrhd_size = 7 # prevent degenerate cases in testing

    # tensor to reduce
    src = torch.randn((batch_size, n_pts, n_channels))

    # randomly generate index pointer tensor for CSR format
    nbrhd_sizes = [torch.tensor([0])]
    while sum(nbrhd_sizes) < n_pts:
        nbrhd_sizes.append(torch.randint(0, max_nbrhd_size + 1, size=(1,)))
        max_nbrhd_size = min(max_nbrhd_size, n_pts - sum(nbrhd_sizes))
    indptr = torch.cumsum(torch.tensor(nbrhd_sizes, dtype=torch.long), dim=0)
    if batch_size == 1:
        src = src.squeeze(0)
    out = segment_csr(src=src, indptr=indptr, reduce='sum')
    
    if batch_size == 1:
        assert out.shape == (len(indptr) - 1, n_channels)
    else:
        assert out.shape == (batch_size, len(indptr) - 1, n_channels)

@pytest.mark.parametrize('batch_size', [1,4])
def test_native_segcsr_reductions(batch_size):
    src = torch.ones([10, 3])
    indptr = torch.tensor([0,3,8,10], dtype=torch.long)

    out_shape = [len(indptr) - 1, src.shape[1]]
    if batch_size > 1:
        src = src.repeat(batch_size, *[1 for _ in range(src.ndim)])
        out_shape = [batch_size] + out_shape
    out_shape = torch.Size(out_shape)

    out_sum = segment_csr(src, indptr, reduce='sum')
    assert out_sum.shape == out_shape
    diff = out_sum - torch.tensor([[3, 5, 2]]).T * torch.ones([3,3])
    assert not diff.nonzero().any()

    out_mean = segment_csr(src, indptr, reduce='mean')
    assert out_mean.shape == out_shape
    diff = out_mean - torch.ones([3,3])
    assert not diff.nonzero().any()
    