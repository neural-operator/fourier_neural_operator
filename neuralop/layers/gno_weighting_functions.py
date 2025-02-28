from functools import partial
import torch

def bump_cutoff(x, radius=1., scale=1., eps=1e-7):
    out = x.clip(0., radius) / radius
    out = - 1 / ((1 - out ** 2) + eps)
    return out.exp() * torch.e * scale

def bump_sqrt_cutoff(x, radius=1., scale=1., eps=1e-7):
    out = - 1 / (1 - x / radius + eps)
    return out.exp() * torch.e * scale

def linear_cutoff(x, radius=1., scale=1.):
    x = (radius - x).clip(0., radius)
    return x * scale / radius

# TODO(jberner): Tanh gives NaNs for the first derivative at 0. and `radius`
def tanh_cutoff(x, radius=1., scale=1., slope=2, eps=1e-6):
    out = x.clip(0., radius) / radius
    out = slope * (2 * out - 1) / (2 * torch.sqrt((1 - out) * out) + eps)
    out = - 0.5 * torch.nn.functional.tanh(out) + 0.5
    return out * scale

def cos_cutoff(x, radius=1., scale=1.):
    x = x / radius
    return scale * (0.5 * torch.cos(torch.pi * x) + 0.5)

def quadr_cutoff(x, radius=1., scale=1.):
    x = x / radius
    left = 1 - 2 * x ** 2
    right = 2 * (1 - x) ** 2
    return scale * torch.where(x < 0.5, left, right)

def cubic_cutoff(x, radius=1., scale=1.):
    b = 3 * scale / (radius ** 2)
    a = 2 * b / (3 * radius)
    out = a * x ** 3 - b * x ** 2 + scale
    assert (x < radius + 0.001).all()
    assert (x > -0.001).all()
    assert (out > -0.001).all()
    return out

def quartic_cutoff(x, radius=1., scale=1.):
    a = scale / radius ** 4
    c = - 2 * scale / radius ** 2 
    return a * x ** 4 + c * x ** 2 + scale

def quartic_sqrt_cutoff(x, radius=1., scale=1.):
    a = scale / radius ** 2
    c = - 2 * scale / radius
    return a * x ** 2 + c * x + scale

def octic_cutoff(x, radius=1., scale=1.):
    x = x / radius
    return scale * (-3 * x ** 8 + 8 * x ** 6 - 6 * x ** 4  + 1)

def octic_sqrt_cutoff(x, radius=1., scale=1.):
    x = x / radius
    return scale * (-3 * x ** 4 + 8 * x ** 3 - 6 * x ** 2  + 1)

def sigmoid_cutoff(x, radius=1., scale=1.):
    denom_1 = torch.e ** (50 * (-radius + 0.1)) + 1
    denom_2 = 1 + torch.e ** (50 * 0.1)
    denom_3 = torch.e ** (50 * (x - radius + 0.1)) + 1

    return scale * 1 / (1 / denom_1 - 1 / denom_2) * (1 / denom_3 - 1 / denom_2)

WEIGHTING_FN_REGISTRY = {
    "linear": linear_cutoff,
    "bump": bump_cutoff,
    "tanh": tanh_cutoff,
    "cubic": cubic_cutoff,
    "cos": cos_cutoff,
    "quadr": quadr_cutoff,
    "bump_sqrt": bump_sqrt_cutoff,
    "quartic": quartic_cutoff,
    "quartic_sqrt": quartic_sqrt_cutoff,
    "octic": octic_cutoff,
    "octic_sqrt": octic_sqrt_cutoff,
    "sigmoid": sigmoid_cutoff
}

def dispatch_weighting_fn(weight_function_name : str, sq_radius: float, scale: float):
    '''
    Select a GNO weighting function for use in output GNO 
    of a Mollified Graph Neural Operator-based model. See [1]_ (add later)

    Parameters
    ----------
    weight_function_name : str Literal
        name of weighting function to use, keyed to ``WEIGHTING_FN_REGISTRY`` above
    sq_radius : float
        squared radius of GNO neighborhoods for Nyström approximation
    scale : float
        factor by which to scale all weights
    '''
    base_func = WEIGHTING_FN_REGISTRY.get(weight_function_name)
    if base_func is None:
        raise NotImplementedError(f"weighting function should be one of {list(WEIGHTING_FN_REGISTRY.keys())}, got {weight_function_name}")
    return partial(base_func, radius=sq_radius, scale=scale)
