import numpy as np
import torch
from diffusionmodel import DiffusionModel
from train import train
from torch.optim import Adam
import bgflow.distribution.sampling.mcmc as MCMC
import bgflow.distribution.energy.double_well as DoubleWell
import bgflow.distribution.normal as Normal
from potential import LinearInterpolation 

torch.random.manual_seed(10)

ts = np.linspace(0, 1, 50)
xs = np.linspace(-5, 5, 50)

X, Y = np.meshgrid(xs, ts)

target = DoubleWell.DoubleWellEnergy(dim=1, b=-0.4, c=0.1)
prior = Normal.NormalDistribution(dim=1)

net = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.SiLU(), torch.nn.Linear(10, 1))

# the total potential that interpolates between target and gaussian
V = LinearInterpolation(prior=prior, target=target, net=net)

sampler = MCMC.GaussianMCMCSampler(energy=target, init_state=torch.tensor([0.]))

# import matplotlib.pyplot as plt
# out = V(torch.from_numpy(X.reshape(50*50, 1)), torch.from_numpy(Y.reshape(50*50, 1)))
# plt.contourf(X, Y, torch.exp(-out.reshape(50, 50)))
# plt.show()

data = sampler.sample(n_samples=1000)

# def plot_hist(x):
#     counts, bins = np.histogram(x.detach().cpu(), bins=np.linspace(-10, 10, 50))
#     plt.stairs(counts, bins, fill=True)
#     plt.show()


# plot_hist(data)


device = 'cuda'
batch_size = 64


diff_model = DiffusionModel(potential=V, device=device)
print(sum([len(p) for p in diff_model.parameters()]))

#plot_sample_noising(diff_model, train_data, n_timesteps = 20, n_samples=3)

train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
optimizer = Adam(diff_model.parameters(), lr=1e-4)


def MSE_loss(prediction, ground_truth):
    loss = ((prediction - ground_truth)**2).mean()
    return loss


train(diff_model, 
      loss_fn=MSE_loss, 
      optimizer=optimizer, 
      data_loader=train_loader, 
      n_iterations=100, 
      device=device, 
      plot_interval=50)
