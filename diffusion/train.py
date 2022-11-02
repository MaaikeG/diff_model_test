import random
import torch
from .diffusionmodel import SDE, DiffusionModel



def default_loss_weighting(t):
    '''a weighting function to weight the loss for each time t. 
    This corresponds to \lambda(t) in eq. (7) of the score-SDE paper.'''
    return 1.


def compute_loss(model : DiffusionModel, sde: SDE, loss_fn:callable, x_0:torch.Tensor, t:torch.FloatType, weighting_fn:callable=default_loss_weighting):
    t_s = torch.full_like(x_0, t)

    x_t = sde.forward_sample(x_0, t_s)

    force = sde.force(x_0, x_t, t_s)
    
    predicted_force, dt = model.force(x_t, t_s)
    
    weight = weighting_fn(t)

    return weight * loss_fn(predicted_force, force) #+ (dt**2).sum()


def batch_iteration(model, sde, loss_fn, optimizer, batch, device):
    model.zero_grad()

    x_0 = batch.to(device)

    t = random.uniform(0, 1)

    loss = compute_loss(model, sde, loss_fn, x_0, t)
    loss.backward()
    optimizer.step()

    return loss.detach()


def train(model, sde, loss_fn, data_loader, optimizer, n_iterations, device='cpu'):
       
    for i in range(n_iterations):

        iteration_loss = 0.

        for batch_idx, batch in enumerate(data_loader):

            loss = batch_iteration(model, sde, loss_fn, optimizer, batch, device)
            loss.detach()

            iteration_loss += loss
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(data_loader)} - batch loss: {loss}', end='\r', flush=True)

        print(f'Epoch {i}/{n_iterations} total loss: {iteration_loss}', end='\n')
        