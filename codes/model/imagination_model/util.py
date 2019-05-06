import importlib
import torch
from addict import Dict

from codes.utils.util import get_product_of_iterable


def get_num_channels_in_image(env_mode):
    if env_mode == "rgb":
        return 3
    else:
        return 1


def get_component(component_name, config):
    return importlib.import_module("codes.model.imagination_model.components.{}".format(component_name)).Model(config)


def get_action_space_size(config):
    return get_product_of_iterable(config.env.action_space.shape)

def merge_first_and_second_dim(batch):
    # Method to modify the shape of a batch by merging the first and the second dimension.
    #  Given a tensor of shape (a, b, c, ...), return a tensor of shape (a*b, c, ...)
    shape = batch.shape
    return batch.view(shape[0] * shape[1], *shape[2:])

def unmerge_first_and_second_dim(batch, first_dim=-1, second_dim=-1):
    # Method to modify the shape of a batch by unmerging the first dimension.
    #  Given a tensor of shape (a*b, c, ...), return a tensor of shape (a, b, c, ...)
    shape = batch.shape
    return batch.view(first_dim, second_dim, *shape[1:])

def sample_zt_from_distribution(mu, logsigma):
    sigma = logsigma.exp()
    eps = torch.randn_like(sigma)
    z_t = eps.mul(sigma).add_(mu)
    to_return = Dict()
    to_return.z_t = z_t
    to_return.mu = mu
    to_return.sigma = sigma
    return to_return

def clamp_mu_logsigma(_mu, _logsigma):
    mu = torch.clamp(_mu, -8., 8.)
    logsigma = torch.clamp(_mu, -1., 1.)
    return mu, logsigma

def gaussian_kld(mu_left, sigma_left, mu_right, sigma_right):
    """
    Provided by @anirudh
    Compute KL divergence between a bunch of univariate Gaussian distributions
    with the given means and log-variances.
    We do KL(N(mu_left, logvar_left) || N(mu_right, logvar_right)).
    """
    logsigma_left = sigma_left.log()
    logsigma_right = sigma_right.log()
    logvar_left = 2 * logsigma_left
    logvar_right = 2 * logsigma_right

    gauss_klds = 0.5 * (logvar_right - logvar_left +
                        (torch.exp(logvar_left) / torch.exp(logvar_right)) +
                        ((mu_left - mu_right) ** 2.0 / torch.exp(logvar_right)) - 1.0)
    assert len(gauss_klds.size()) == 2
    return gauss_klds
    return torch.sum(gauss_klds, 1)