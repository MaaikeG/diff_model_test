import torch
import random
import math
from numpy import ndarray


def make_betas(T, variance_schedule):
    if isinstance(variance_schedule, torch.Tensor):
        return variance_schedule
    if isinstance(variance_schedule, ndarray):
        return torch.from_numpy(variance_schedule)
    if variance_schedule == 'linear':
        return torch.linspace(1e-4, 0.03, T)
    if variance_schedule == 'cosine':
        NotImplemented


def get_items_reshaped(arr, indices, shape):
    items = arr[indices]
    while len(shape) > items.ndim:
        items = items[..., None]
    items = torch.broadcast_to(items, shape)
    return items


class DiffusionModel(torch.nn.Module):
    def __init__(self, net, T=100, variance_schedule='Linear', 
                 double_precision=False, 
                 device='cuda', 
                 *args):
        super().__init__()

        self.device = device
        betas = make_betas(T, variance_schedule).to(device)

        if double_precision:
            self.betas = betas.double()
            self.net = net.double()
        else:
            self.betas = betas.float()
            self.net = net.float()

        # initialize all combinations of betas we need here
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, 0)
        self.one_minus_alpha_cumprod = 1 - self.alpha_cum_prod
        
        self.sqrt_alpha_inverse = 1 / torch.sqrt(self.alphas)

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(self.one_minus_alpha_cumprod)


        self.to(device)

        
    @property
    def T(self):
        return len(self.betas)


    def forward(self, x_t, t_s):
        return self.net(torch.hstack([x_t, t_s]))
        

    def apply_noise(self, x_0, t_s):
        noise = torch.randn_like(x_0)

        mean_mult = get_items_reshaped(self.sqrt_alpha_cumprod, t_s, x_0.shape) 
        mean = mean_mult * x_0 
        
        variance_mult = get_items_reshaped(self.sqrt_one_minus_alpha_cumprod, t_s, x_0.shape)
        variance = variance_mult * noise
        
        noisy_x = mean + variance
        return noisy_x, noise


    def loss(self, x_0, loss_fn):
    
        t = random.randint(0, self.T-1)
        t_s = torch.full([x_0.shape[0]], t).to(x_0.device)  

        x_t, noise = self.apply_noise(x_0, t_s)

        predicted_noise = self.forward(x_t, t_s)

        return loss_fn(predicted_noise, noise)


    def sample(self, shape):
        # start with random noise
        x_t = torch.randn(size=shape, device=self.device)

        with torch.no_grad():
            for t in reversed(range(1, self.T)):
                # make an array of t_s, one for each sample we want.
                t_s = torch.full_like(x_t, fill_value=t, device=self.device)

                predicted_noise = self.forward(x_t, t_s)
                mean = self.sqrt_alpha_inverse[t] * \
                    (x_t - predicted_noise * self.betas[t] / self.sqrt_one_minus_alpha_cumprod[t]) 

                noise = torch.randn_like(x_t)
                x_t = mean + math.sqrt(self.betas[t]) * noise
            
        return x_t