import time

import numpy as np
import torch
import torch.nn as nn

from typing import Any

from matplotlib import pyplot as plt

from fbpinns import losses
from fbpinns.plot import plot_main
from fbpinns.common.utils import _x_random, _x_mesh
from fbpinns.plot.plot_main import plot_pinn_simulation
from fbpinns.trainers_base import _Trainer
from fbpinns.constants import Constants

import logging

logger = logging.getLogger(__name__)


class PINNTrainer(_Trainer):
    """Standard PINN model trainer class"""

    def _train_step(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            c: Constants, idx: int,
            mu: float, sd: float, device: str
    ) -> tuple[Any, ...]:  # use separate function to ensure computational graph/memory is released

        optimizer.zero_grad()
        model.train()

        sampler = _x_random if c.RANDOM else _x_mesh
        input_x = sampler(c.SUBDOMAIN_XS, c.BATCH_SIZE, device).requires_grad_(True)  # here we get our collocation points
        u = model((input_x-mu)/sd)  # normalise before passing to model
        u = u*c.Y_N[1] + c.Y_N[0]  # post process with mean and standard deviation

        # get gradients
        uj = c.P.get_gradients(input_x, u)  # problem-specific

        # apply hard boundary constraints
        uj = c.P.boundary_condition(input_x, *uj, *c.BOUNDARY_N)   # problem-specific
        # uj = c.P.boundary_condition(input_x, *uj, sd=c.BOUNDARY_N)   # problem-specific

        # backprop loss
        loss = c.P.physics_loss(input_x, *uj)  # problem-specific

        loss.backward()
        optimizer.step()

        # if (idx % c.SUMMARY_FREQ) == 0:
        #     print(*[t.shape for t in uj], input_x.shape)


        # return result
        return input_x.detach(), [t.detach() for t in uj], loss.item()


    def train(self):
        "Train model"

        c, device, writer = self.c, self.device, self.writer

        # define model using dims (in: 3, out = 1, hidden and num layers)
        model = c.MODEL(c.P.d[0], c.P.d[1], c.N_HIDDEN, c.N_LAYERS)  # problem-specific

        # create optimiser
        optimizer = torch.optim.Adam(model.parameters(), lr=c.LRATE)

        # put model on device
        model.to(device)

        """
        Get normalisation values, most likely (tensor([-1., -1.,  0.]), tensor([1., 1., 1.]))
        As we are not using the domain decomposition, 
         we are just taking the min and max values across the entire domain
        """
        xmin, xmax = torch.tensor([[x.min(), x.max()] for x in c.SUBDOMAIN_XS], dtype=torch.float32, device=device).T

        mu = (xmin + xmax)/2  # most likely (0, 0, 0)
        sd = (xmax - xmin)/2  # and (1, 1, 1)

        logger.info(f"{mu}, {sd}, {mu.shape}, {sd.shape}")

        # get exact solution if it exists
        x_test = _x_mesh(c.SUBDOMAIN_XS, c.BATCH_SIZE_TEST, device)
        yj_true = c.P.exact_solution(x_test, c.BATCH_SIZE_TEST)# problem-specific

        ## TRAIN

        mstep, fstep, yj_test_losses = 0, 0, []
        start, gpu_time = time.time(), 0.
        for i in range(c.N_STEPS):
            gpu_start = time.time()
            x, yj, loss = self._train_step(model, optimizer, c, i, mu, sd, device)
            mstep += model.size  # record number of weights updated
            fstep += model.flops(x.shape[0])  # record number of FLOPS
            gpu_time += time.time()-gpu_start


            # METRICS

            if (i + 1) % c.SUMMARY_FREQ == 0:

                # set counters
                rate, gpu_time = c.SUMMARY_FREQ / gpu_time, 0.

                # print summary
                self._print_summary(i, loss, rate, start)

                # test step
                yj_test_losses = self._test_step(x_test, yj_true,   x, yj,   model, c, i, mstep, fstep, writer, yj_test_losses)

            # SAVE

            if (i + 1) % c.MODEL_SAVE_FREQ == 0:

                # save model and losses
                self._save_model(i, model)
                np.save(c.MODEL_OUT_DIR+"loss_%.8i.npy"%(i + 1), np.array(yj_test_losses))

        # cleanup
        writer.close()
        print("Finished training")


    def _test_step(self, x_test, yj_true,   x, yj,   model, c, i, mstep, fstep, writer, yj_test_losses):# use separate function to ensure computational graph/memory is released

        # get full model solution using test data
        yj_full, y_full_raw = full_model_PINN(x_test, model, c)
        # print(x_test.shape, yj_true[0].shape, yj_full[0].shape)

        # get losses over test data
        yj_test_loss = [losses.l1_loss(a,b).item() for a,b in zip(yj_true, yj_full)]
        physics_loss = c.P.physics_loss(x_test, *yj_full).item()# problem-specific
        yj_test_losses.append([i + 1, mstep, fstep]+yj_test_loss+[physics_loss])
        for j,l in enumerate(yj_test_loss):
            for step,tag in zip([i + 1, mstep, fstep], ["istep", "mstep", "zfstep"]):
                writer.add_scalar("loss_%s/yj%i/test"%(tag,j), l, step)
        writer.add_scalar("loss_istep/zphysics/test", physics_loss, i + 1)

        # PLOTS

        if (i + 1) % c.TEST_FREQ == 0:

            # save figures
            # fs = plot_main.plot_PINN(x_test, yj_true, x, yj, yj_full, y_full_raw, yj_test_losses, c, i + 1)
            # if fs is not None: self._save_figs(i, fs)
            # print(yj_full.shape)
            uhat = yj_full[0].reshape(c.BATCH_SIZE_TEST).detach().cpu()
            plot_pinn_simulation(uhat=uhat, dt=5, plot_diff=False, standardise=False, cmap='viridis')
            plt.show()

        del x_test, yj_true,   x, yj,   yj_full, y_full_raw# fixes weird over-allocation of GPU memory bug caused by plotting (?)

        return yj_test_losses


def full_model_PINN(x, model, c):
    """Get the full PINN prediction (forward inference only)"""

    # get normalisation values
    xmin, xmax = torch.tensor([[x.min(), x.max()] for x in c.SUBDOMAIN_XS], dtype=torch.float32, device=x.device).T
    mu = (xmin + xmax)/2; sd = (xmax - xmin)/2

    # get full model solution using test data
    x_ = x.detach().clone().requires_grad_(True)
    y = model((x_-mu)/sd)
    y_raw = y.detach().clone()
    y = y*c.Y_N[1] + c.Y_N[0]

    # get gradients
    yj = c.P.get_gradients(x_, y)# problem-specific

    # detach from graph
    yj = [t.detach() for t in yj]

    # apply boundary conditions
    # yj = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
    yj = c.P.boundary_condition(x, *yj, sd=c.BOUNDARY_N)# problem-specific

    return yj, y_raw
