from turtle import forward
import torch
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



class SinusodalEmbedding(torch.nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        # we'll be concatenating sin and cos for half_dim elements, so that
        # the total embedding is length embedding_size
        half_size = embedding_size // 2

        divisor = - math.log(10000) / half_size
        self.divisor = torch.exp(torch.arange(half_size) * divisor)

    def forward(self, x):
        embeddings = x[:, None] * self.divisor[None,:].to(x.device)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


def construct_embedding_MLP(embedding_size):
    return torch.nn.Sequential(
            SinusodalEmbedding(embedding_size),
            torch.nn.Linear(embedding_size, embedding_size), 
            torch.nn.ReLU()
        )


class DiffusionModel(torch.nn.Module):
    def __init__(self, net, T=100, variance_schedule='Linear', 
                 double_precision=False, 
                 device='cuda', 
                 time_embedding_size=32,
                 use_guidance=False,
                 class_embedding_size=32,
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

        self.time_MLP = construct_embedding_MLP(time_embedding_size)
                
        self.use_guidance = use_guidance
        if use_guidance:
            self.class_MLP = construct_embedding_MLP(class_embedding_size)
        else:
            self.class_MLP = None


        self.to(device)

        
    @property
    def T(self):
        return len(self.betas)


    def forward(self, x_t, t_s, classes=None):
        
        time_embedding = self.time_MLP(t_s)

        if self.use_guidance:
            class_embedding = self.class_MLP(classes)
            predicted_noise = self.net(x_t, time_embedding, class_embedding) 
            
        else:
            predicted_noise = self.net(x_t, time_embedding) 
        
        return predicted_noise


    def apply_noise(self, x_0, t_s):
        noise = torch.randn_like(x_0)

        mean_mult = get_items_reshaped(self.sqrt_alpha_cumprod, t_s, x_0.shape) 
        mean = mean_mult * x_0 
        
        variance_mult = get_items_reshaped(self.sqrt_one_minus_alpha_cumprod, t_s, x_0.shape)
        variance = variance_mult * noise
        
        noisy_x = mean + variance
        return noisy_x, noise


    def sample(self, shape):
        # start with random noise
        x_t = torch.randn(size=shape, device=self.device)

        with torch.no_grad():
            for t in reversed(range(1, self.T)):
                # make an array of t_s, one for each sample we want.
                t_s = torch.full(size=[shape[0]], fill_value=t, device=self.device)

                predicted_noise = self.forward(x_t, t_s)
                mean = self.sqrt_alpha_inverse[t] * \
                    (x_t - predicted_noise * self.betas[t] / self.sqrt_one_minus_alpha_cumprod[t]) 

                noise = torch.randn_like(x_t)
                x_t = mean + math.sqrt(self.betas[t]) * noise
            
        return x_t