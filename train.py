from random import random
import random
import torch
import matplotlib.pyplot as plt
from plotting import plot_samples


def get_loss(model, loss_fn, x_0, t, dt):

    d_x = model.apply_noise(x_0, t, dt)

    prediction = model.forward(x_t, t_s)

    return loss_fn(prediction, ground_truth)


def batch_iteration(model, loss_fn, optimizer, batch, device):
    model.zero_grad()

    x_0 = batch[0].to(device)

    t = random.uniform(0, 1)
    dt = abs(random.normalvariate(0, 0.001))

    # ensure we don't go out of bounds for t
    if t + dt >= 1:
        t -= (t + dt - 1)

    loss = get_loss(model, loss_fn, x_0, t, dt)
    loss.backward()
    optimizer.step()

    return loss.detach()


def train(model, loss_fn, data_loader, optimizer, n_iterations, scheduler=None, device='cpu', plot_interval=100):
       
    for i in range(n_iterations):

        if plot_interval >= 0 and i % plot_interval == 0:        
            plot_samples(model, shape = data_loader.dataset[0][0].shape, n_samples = 10, device=device)


        iteration_loss = 0.

        for batch_idx, batch in enumerate(data_loader):

            loss = batch_iteration(model, loss_fn, optimizer, batch, device)
            loss.detach()

            iteration_loss += loss
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(data_loader)} - batch loss: {loss}', end='\r', flush=True)


    
        if scheduler is not None:
            scheduler.step(iteration_loss)

        print(f'Epoch {i}/{n_iterations} total loss: {iteration_loss}', end='\n')
        