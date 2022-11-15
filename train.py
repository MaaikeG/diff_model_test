def batch_iteration(model, x_0, loss_fn, optimizer):
    model.zero_grad()

    loss = model.loss(x_0, loss_fn)
    loss.backward()
    optimizer.step()

    return loss.detach()


def train(model, loss_fn, data_loader, optimizer, n_iterations, scheduler=None, device='cpu', callback_interval=-1, callback=None):
       
    for i in range(n_iterations):

        if callback_interval >= 0 and i % callback_interval == 0:    
            callback(model)

        iteration_loss = 0.

        for batch_idx, batch in enumerate(data_loader):

            loss = batch_iteration(model, batch[0].to(device), loss_fn, optimizer)
            loss.detach()

            iteration_loss += loss
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(data_loader)} - batch loss: {loss}', end='\r', flush=True)
    
        if scheduler is not None:
            scheduler.step(iteration_loss)

        print(f'Epoch {i}/{n_iterations} total loss: {iteration_loss}', end='\n')
        