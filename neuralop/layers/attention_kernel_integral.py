import torch
from torch import nn
import math
from torch.nn.init import xavier_uniform_, zeros_


class AttentionKernelIntegral(torch.nn.Module):
    """
    Kernel integral transform with attention
    Computes \int_{Omega} k(x, y) * f(y) dy,
    where:
          K(x, y) = \sum_{c=1}^d q_c(x) * k_c(y), q(x) = [q_1(x); ...; q_d(x)], k(y) = [k_1(y); ...; k_d(y)]
          f(y) = v(y)
    More specifically, this module supports using just one input function (self-attention) or
    two input functions (cross-attention) to compute the kernel integral transform.

    1. Self-attention:
        input function u(.), sampling grid D_x = {x_i}_{i=1}^N
        query function: q(x_i) = u(x_i) W_q
        key function: k(x_i) = u(x_i) W_k
        value function: v(x_i) = u(x_i) W_v

    2. Cross-attention:
        first input function u_qry(.), sampling grid D_x = {x_i}_{i=1}^N
        second input function u_src(.), sampling grid D_y = {y_j}_{j=1}^M, D_y can be different from D_x
        query function: q(x_i) = u_qry(x_i) W_q
        key function: k(y_j) = u_src(y_j) W_k
        value function: v(y_j) = u_src(y_j) W_v

    Self-attention can be considered as a special case of cross-attention, where u = u_qry = u_src and D_x = D_y.

    The kernel integral transform will be numerically computed as:
        \int_{Omega} k(x, y) * f(y) dy \appox \sum_{j=1}^M * k(x, y_j) * f(y_j) * w(y_j)
    For uniform quadrature, the weights w(y_j) = 1/M.
    For non-uniform quadrature, the weights w(y_j) is specified as an input to the forward function.

    Parameters
    ----------
    in_channels : int, input channels
    out_channels : int, output channels
    n_heads : int, number of attention heads in multi-head attention
    head_n_channels : int, dimension of each attention head, determines how many function bases to use for the kernel
                      k(x, y) = \sum_{c=1}^d \q_c(x) * \k_c(y), head_n_channels controls the d
    pos_dim : int, dimension of the domain, determines the dimension of coordinates
    project_query : bool, whether to project the query function with pointwise linear layer
                   (this is sometimes not needed when using cross-attention)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_heads,
                 head_n_channels,
                 project_query=True,
                 ):
        super().__init__()
        self.n_heads = n_heads
        self.head_n_channels = head_n_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.project_query = project_query
        if project_query:
            self.to_q = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)
        else:
            self.to_q = nn.Identity()

        self.to_k = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)

        self.k_norm = nn.InstanceNorm1d(head_n_channels, affine=False)
        self.v_norm = nn.InstanceNorm1d(head_n_channels, affine=False)

        self.to_v = nn.Linear(in_channels, head_n_channels * n_heads, bias=False)

        self.to_out = nn.Linear(head_n_channels * n_heads, out_channels) \
            if head_n_channels * n_heads != out_channels else nn.Identity()

        self.init_gain = 1 / math.sqrt(head_n_channels)
        self.diagonal_weight = self.init_gain
        self.initialize_qkv_weights()

    def init_weight(self, weight, init_fn):
        """
        Initialization for the projection matrix
        basically initialize the weights for each heads with predefined initialization function and gain,
        to add the diagonal bias, it requires input channels = head_n_channels
        W = init_fn(W) + I * diagonal_weight

        init_fn is xavier_uniform_ by default
        """

        for param in weight.parameters():
            if param.ndim > 1:
                for h in range(self.n_heads):
                    init_fn(param[h * self.head_n_channels:(h + 1) * self.head_n_channels, :], gain=self.init_gain)
                    if self.head_n_channels == self.in_channels:
                        diagonal_bias = self.diagonal_weight * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
                        param.data[h * self.head_n_channels:(h + 1) * self.head_n_channels, :] += diagonal_bias

    def initialize_qkv_weights(self):
        """
        Initialize the weights for q, k, v projection matrix with a small gain and add a diagonal bias,
        this technique has been found useful for scale-sensitive problem that has not been normalized
        see Table 8 in https://arxiv.org/pdf/2105.14995.pdf
        """
        init_fn = xavier_uniform_

        if self.project_query:
            self.init_weight(self.to_q, init_fn)
        self.init_weight(self.to_k, init_fn)
        self.init_weight(self.to_v, init_fn)

    def normalize_wrt_domain(self, u, norm_fn):
        """
        Normalize the input function with respect to the domain,
         reshape the tensor to [batch_size*n_heads, num_grid_points, head_n_channels]
        The second dimension is equal to the number of grid points that discretize the domain
        """
        # u: the input or transformed function
        batch_size = u.shape[0]
        u = u.view(batch_size*self.n_heads, -1, self.head_n_channels)
        if isinstance(norm_fn, nn.InstanceNorm1d) or isinstance(norm_fn, nn.GroupNorm):
            u = u.permute(0, 2, 1).contiguous()
        u = norm_fn(u)    # layer norm with channel dimension or instance norm with spatial dimension
        if isinstance(norm_fn, nn.InstanceNorm1d) or isinstance(norm_fn, nn.GroupNorm):
            u = u.permute(0, 2, 1).contiguous()
        return u.view(batch_size, self.n_heads, -1, self.head_n_channels)

    def forward(self,
                u_src,
                pos_src,
                positional_embedding_module=None,    # positional encoding module for encoding q/k
                u_qry=None,
                pos_qry=None,
                weights=None,
                associative=True,   # can be much faster if num_grid_points is larger than the channel number c
                return_kernel=False):
        """
        Computes kernel integral transform with attention

        Parameters
        ----------
        u_src: input function (used to compute key and value in attention),
                tensor of shape [batch_size, num_grid_points_src, channels]
        pos_src: coordinate of the second source of function's sampling points y,
                tensor of shape [batch_size, num_grid_points_src, pos_dim]
        positional_embedding_module: positional embedding module for encoding query/key (q/k),
                a torch.nn.Module
        u_qry: query function,
                tensor of shape [batch_size, num_grid_points_query, channels], if not provided, u_qry = u_src
        pos_qry: coordinate of query points x,
                tensor of shape [batch_size, num_grid_points_query, pos_dim], if not provided, pos_qry = pos_src
        weights : quadrature weight w(y_j) for the kernel integral: u(x_i) = sum_{j} k(x_i, y_j) f(y_i) w(y_j),
                tensor of shape [batch_size, num_grid_points_src], if not provided assume to be 1/num_grid_points_src
        associative: if True, use associativity of matrix multiplication, first multiply K^T V, then multiply Q,
                much faster when num_grid_points is larger than the channel number (which is usually the case)
        return_kernel: if True, return the kernel matrix (for analyzing the kernel)

        Output
        ----------
        u: Output function given on the query points x.
        """

        if u_qry is None:
            u_qry = u_src   # go back to self attention
            if pos_qry is not None:
                raise ValueError('Query coordinates are provided but query function is not provided')
        else:
            if pos_qry is None:
                raise ValueError('Query coordinates are required if query function is provided')

        if return_kernel and associative:
            raise ValueError('Cannot get kernel matrix when associative is set to True')

        batch_size, num_grid_points = u_src.shape[:2]   # batch size and number of grid points
        pos_dim = pos_src.shape[-1]   # position dimension

        q = self.to_q(u_qry)
        k = self.to_k(u_src)
        v = self.to_v(u_src)

        q = q.view(batch_size, -1, self.n_heads, self.head_n_channels).permute(0, 2, 1, 3).contiguous()
        k = k.view(batch_size, -1, self.n_heads, self.head_n_channels).permute(0, 2, 1, 3).contiguous()
        v = v.view(batch_size, -1, self.n_heads, self.head_n_channels).permute(0, 2, 1, 3).contiguous()

        k = self.normalize_wrt_domain(k, self.k_norm)
        v = self.normalize_wrt_domain(v, self.v_norm)

        if positional_embedding_module is not None:
            if pos_dim == 2:
                k_freqs_1 = positional_embedding_module.forward(pos_src[..., 0])
                k_freqs_2 = positional_embedding_module.forward(pos_src[..., 1])
                k_freqs_1 = k_freqs_1.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                k_freqs_2 = k_freqs_2.unsqueeze(1).repeat([1, self.n_heads, 1, 1])

                if pos_qry is None:
                    q_freqs_1 = k_freqs_1
                    q_freqs_2 = k_freqs_2
                else:
                    q_freqs_1 = positional_embedding_module.forward(pos_qry[..., 0])
                    q_freqs_2 = positional_embedding_module.forward(pos_qry[..., 1])
                    q_freqs_1 = q_freqs_1.unsqueeze(1).repeat([1, self.n_heads, 1, 1])
                    q_freqs_2 = q_freqs_2.unsqueeze(1).repeat([1, self.n_heads, 1, 1])

                q = positional_embedding_module.apply_2d_rotary_pos_emb(q, q_freqs_1, q_freqs_2)
                k = positional_embedding_module.apply_2d_rotary_pos_emb(k, k_freqs_1, k_freqs_2)
            elif pos_dim == 1:

                k_freqs = positional_embedding_module.forward(pos_src[..., 0])
                k_freqs = k_freqs.unsqueeze(1).repeat([batch_size, self.n_heads, 1, 1])

                if pos_qry is None:
                    q_freqs = k_freqs
                else:
                    q_freqs = positional_embedding_module.forward(pos_qry[..., 0])
                    q_freqs = q_freqs.unsqueeze(1).repeat([batch_size, self.n_heads, 1, 1])

                q = positional_embedding_module.apply_1d_rotary_pos_emb(q, q_freqs)
                k = positional_embedding_module.apply_1d_rotary_pos_emb(k, k_freqs)
            else:
                raise ValueError('Currently doesnt support relative embedding >= 3 dimensions')

        if weights is not None:
            weights = weights.view(batch_size, 1, num_grid_points, 1)
        else:
            weights = 1.0 / num_grid_points

        if associative:
            dots = torch.matmul(k.transpose(-1, -2), v)
            u = torch.matmul(q, dots) * weights
        else:
            # this is more efficient when num_grid_points<<channels
            kxy = torch.matmul(q, k.transpose(-1, -2))
            u = torch.matmul(kxy, v) * weights

        u = u.permute(0, 2, 1, 3).contiguous().view(batch_size, num_grid_points, self.n_heads*self.head_n_channels)
        u = self.to_out(u)
        if return_kernel:
            return u, kxy
        return u

