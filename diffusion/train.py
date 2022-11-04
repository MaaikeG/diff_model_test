from typing import Iterable, Callable
import random
import torch
from .diffusionmodel import DiffusionModel



def _batch_iteration(model: DiffusionModel,
                    loss_fn_dx: Callable,
                    loss_fn_dt: Callable,
                    optimizer: torch.optim.Optimizer,
                    batch: torch.Tensor,
                    device: torch.DeviceObjType) -> torch.Tensor:
    model.zero_grad()

    x_0 = batch.to(device)

    t = random.uniform(0, 1)

    loss = model.loss(x_0, t, loss_fn_dx, loss_fn_dt)
    loss.backward()
    optimizer.step()

    return loss.detach()


def train(model: DiffusionModel,
          loss_fn_dx: Callable,
          loss_fn_dt: Callable,
          data_loader: Iterable,
          optimizer: torch.optim.Optimizer,
          n_iterations: int,
          device: torch.DeviceObjType = 'cpu'):
    '''Train a model'''
    for i in range(n_iterations):

        iteration_loss = 0.

        for batch_idx, batch in enumerate(data_loader):

            loss = _batch_iteration(
                model, loss_fn_dx, loss_fn_dt, optimizer, batch, device)
            loss.detach()

            iteration_loss += loss

            if batch_idx % 10 == 0:
                print(
                    f'Batch {batch_idx}/{len(data_loader)} - batch loss: {loss}', end='\r', flush=True)

        print(
            f'Epoch {i}/{n_iterations} total loss: {iteration_loss}', end='\n')
