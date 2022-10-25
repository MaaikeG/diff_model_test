from random import random
import random
import torch
import matplotlib.pyplot as plt
from plotting import plot_samples


def get_loss(model, loss_fn, x_0, t_s, class_encoding):

    x_t, ground_truth = model.apply_noise(x_0, t_s)

    prediction = model.forward(x_t, t_s, class_encoding)

    return loss_fn(prediction, ground_truth)


def batch_iteration(model, loss_fn, optimizer, batch, device):
    model.zero_grad()

    x_0 = batch[0].to(device)

    class_encoding = torch.nn.functional.one_hot(batch[1], num_classes = 10).to(device)

    t = random.randint(0, model.T-1)
    t_s = torch.full([x_0.shape[0]], t).to(x_0.device)  

    loss = get_loss(model, loss_fn, x_0, t_s, class_encoding)
    loss.backward()
    optimizer.step()

    return loss.detach()


def train(model, loss_fn, data_loader, optimizer, n_iterations, scheduler=None, device='cpu', plot_interval=100):
       
    for i in range(n_iterations):


        iteration_loss = 0.

        for batch_idx, batch in enumerate(data_loader):

            loss = batch_iteration(model, loss_fn, optimizer, batch, device)
            loss.detach()

            iteration_loss += loss
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(data_loader)} - batch loss: {loss}', end='\r', flush=True)


        if True:#plot_interval >= 0 and i % plot_interval == 0:        
            plot_samples(model, shape = data_loader.dataset[0][0].shape, n_samples = 10, device=device)

    
        if scheduler is not None:
            scheduler.step(iteration_loss)

        print(f'Epoch {i}/{n_iterations} total loss: {iteration_loss}', end='\n')
        