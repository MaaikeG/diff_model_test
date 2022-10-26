import torch
import matplotlib.pyplot as plt
from diffusionmodel import SinusodalEmbedding


def plot_samples(model, shape, n_samples = 10):
    shape = [n_samples] + list(shape) 
    samples = model.sample(shape)
    _, axes = plt.subplots(1, 10, figsize=(20, 2))
    for ax_idx, ax in enumerate(axes):
        ax.contourf(samples[ax_idx][0].cpu())
    plt.show()


def plot_sample_noising(model, samples, n_timesteps = 20, n_samples=3):
    # NOISING THE SAMPLE WORKS!
    import matplotlib.pyplot as plt

    # make a grid for the samples and all the noise levels
    _, axes = plt.subplots(n_samples, n_timesteps, figsize=(n_timesteps, n_samples))
    # get the first few samples from the training set
    samples = torch.stack([samples[s][0] for s in range(n_samples)]).cuda() # can't index a range in the dataset so do it like this

    # for each timestep (=column in the subplot grid), noise the samples to that t and plot the noised samples
    for ax_column, t in zip(axes.T, torch.linspace(0, model.T-1, n_timesteps).int()):
        t_s = torch.full([samples.shape[0]], t).to(samples.device)  

        with torch.no_grad():
            noisy_samples, _ = model.apply_noise(samples, t_s)
            for s in range(n_samples):
                ax_column[s].contourf(noisy_samples[s][0].cpu())
    plt.show()


def plot_time_embeddings():
    embedder = SinusodalEmbedding(32)

    t_s = torch.tensor(range(300))
    embedding = embedder.forward(t_s)
    plt.contourf(embedding)
    plt.show()

