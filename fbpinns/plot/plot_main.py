#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:33:12 2021

@author: bmoseley
"""

# This module imports and calls various plotting functions depending on the dimensionality of the FBPINN / PINN problem

# This module is used during training by main.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from fbpinns.plot import plot_main_2D, plot_main_1D, plot_main_3D


def plot_FBPINN(*args):
    "Generates FBPINN plots during training"
    
    # figure out dimensionality of problem, use appropriate plotting function
    c = args[9]
    nd = c.P.d[0]
    if   nd == 1:
        return plot_main_1D.plot_1D_FBPINN(*args)
    elif nd == 2:
        return plot_main_2D.plot_2D_FBPINN(*args)
    elif nd == 3:
        return plot_main_3D.plot_3D_FBPINN(*args)
    else:
        return None
        # TODO: implement higher dimension plotting


def plot_PINN(*args):
    "Generates PINN plots during training"
    
    # figure out dimensionality of problem, use appropriate plotting function
    c = args[7]
    nd = c.P.d[0]
    if   nd == 1:
        return plot_main_1D.plot_1D_PINN(*args)
    elif nd == 2:
        return plot_main_2D.plot_2D_PINN(*args)
    elif nd == 3:
        return plot_main_3D.plot_3D_PINN(*args)
    else:
        return None
        # TODO: implement higher dimension plotting


def plot_pinn_simulation(
        uhat: torch.Tensor, u: torch.Tensor | None = None, dt: int = 10, plot_diff: bool = False, standardise: bool = False, cmap: str = "seismic"
) -> None:

    vmin, vmax = None, None
    if standardise and u is not None:
        vmin, vmax = u.min().item(), u.max().item()

    indices = np.arange(0, uhat.shape[0], dt)
    rows = 1 if u is None else 2
    rows = rows if not plot_diff else rows + 1
    fig, axes = plt.subplots(
        rows, len(indices), figsize=(len(indices) * 5, 5 * rows)
    )

    for idx, i in enumerate(indices):
        if u is not None:
            axes[0, idx].set_title(f"Numerical: time step = {i}")
            axes[0, idx].imshow(u[:, :, i].T, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1, idx].set_title("PINN")
            axes[1, idx].imshow(uhat[:, :, i].T, cmap=cmap, vmin=vmin, vmax=vmax)
            if plot_diff:
                axes[2, idx].set_title("Difference")
                axes[2, idx].imshow(uhat[:, :, i].T - u[:, :, i].T, cmap='gray', vmin=vmin, vmax=vmax)
        else:
            axes[idx].set_title(f"PINN: time step = {i}")
            axes[idx].imshow(uhat[:, :, i].T, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.show()
