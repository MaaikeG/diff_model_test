from random import random
import random
import torch
import matplotlib.pyplot as plt
from plotting import plot_samples


def get_loss(model, loss_fn, x_0, t_s):
    x_t, noise = model.apply_noise(x_0, t_s)

    predicted_noise = model.forward(x_t, t_s)

    return loss_fn(predicted_noise, noise)


def batch_iteration(model, loss_fn, optimizer, x_0):
    model.zero_grad()

    t = random.randint(0, model.T-1)
    t_s = torch.full([x_0.shape[0]], t).to(x_0.device)  

    loss = get_loss(model, loss_fn, x_0, t_s)
    loss.backward()
    optimizer.step()

    return loss.detach()


def train(model, loss_fn, data_loader, optimizer, n_iterations, scheduler=None, device='cpu', callback_interval=-1, callback=None):
       
    for i in range(n_iterations):

        if callback_interval >= 0 and i % callback_interval == 0:    
            callback(model)

        iteration_loss = 0.

        for batch_idx, batch in enumerate(data_loader):

            loss = batch_iteration(model, loss_fn, optimizer, batch[0].to(device))
            loss.detach()

            iteration_loss += loss
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(data_loader)} - batch loss: {loss}', end='\r', flush=True)
    
        if scheduler is not None:
            scheduler.step(iteration_loss)

        print(f'Epoch {i}/{n_iterations} total loss: {iteration_loss}', end='\n')
        