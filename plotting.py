import torch
import matplotlib.pyplot as plt


def plot_samples(model, shape, n_samples = 10, classes = torch.as_tensor([0,1,2,3,4,5,6,7,8,9]), device='cpu'):
    shape = [n_samples] + list(shape) 
    class_encoding = torch.nn.functional.one_hot(classes, num_classes = 10).to(device)
    samples = model.sample(shape, classes=class_encoding, w=3.)
    _, axes = plt.subplots(1, 10, figsize=(20, 2))
    for ax_idx, ax in enumerate(axes):
        ax.contourf(samples[ax_idx][0].cpu(), origin='upper')
    plt.show()


def plot_sample_noising(model, samples, n_timesteps = 20, n_samples=3, device='cpu'):
    # NOISING THE SAMPLE WORKS!
    import matplotlib.pyplot as plt

    # make a grid for the samples and all the noise levels
    _, axes = plt.subplots(n_samples, n_timesteps, figsize=(n_timesteps, n_samples))
    # get the first few samples from the training set
    samples = torch.stack([samples[s][0] for s in range(n_samples)]).to(device) # can't index a range in the dataset so do it like this

    # for each timestep (=column in the subplot grid), noise the samples to that t and plot the noised samples
    for ax_column, t in zip(axes.T, torch.linspace(0, model.T-1, n_timesteps).int()):
        t_s = torch.full([samples.shape[0]], t).to(device)  

        with torch.no_grad():
            noisy_samples, _ = model.apply_noise(samples, t_s)
            for s in range(n_samples):
                ax_column[s].contourf(noisy_samples[s][0].cpu())
    plt.show()


def plot_time_embeddings():
    from unet import SinusodalEmbedding

    embedder = SinusodalEmbedding(32)

    t_s = torch.tensor(range(300))
    embedding = embedder.forward(t_s)
    plt.contourf(embedding)
    plt.show()

