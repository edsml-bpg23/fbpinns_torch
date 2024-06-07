import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def init_logging() -> None:
    """
    Initialise the logging configuration.
    """
    logging.basicConfig(
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        force=True,
    )


def get_device() -> str:
    """
    Determine the available computing device (CPU or GPU).

    Returns
    -------
    str
        The name of the device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        logger.info("GPU activated.")
        device = "cuda"
    else:
        logger.info("No GPU available, CPU activated.")
        device = "cpu"
    return device


def _x_random(subdomain_xs, batch_size, device):
    """Get flattened random samples of x"""
    s = torch.tensor([[x.min(), x.max()] for x in subdomain_xs], dtype=torch.float32, device=device).T.unsqueeze(1)# (2, 1, nd)
    x_random = s[0]+(s[1]-s[0])*torch.rand((np.prod(batch_size), len(subdomain_xs)), device=device)# random samples in domain
    return x_random


def _x_mesh(subdomain_xs, batch_size, device):
    """Get flattened samples of x on a mesh"""
    x_mesh = [torch.linspace(x.min(), x.max(), b, device=device) for x, b in zip(subdomain_xs, batch_size)]
    x_mesh = torch.stack(torch.meshgrid(*x_mesh, indexing="ij"), -1).view(-1, len(subdomain_xs))# nb torch.meshgrid uses np indexing="ij"
    return x_mesh
