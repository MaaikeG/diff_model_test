from ast import arg
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



class DiffusionModel(torch.nn.Module):
    def __init__(self, net, T=100, variance_schedule='Linear', 
                 double_precision=False, 
                 device='cuda', 
                 class_dropout=0.5,
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

        self.condition_information_dropout = torch.nn.Dropout(p = class_dropout)

        self.to(device)

        
    @property
    def T(self):
        return len(self.betas)


    def forward(self, x_t, t_s, classes):
        # perform dropout on the conditional information so we don't rely on only the class
        classes = self.condition_information_dropout(classes.float())
        return self.net(x_t, t_s, classes)
                     

    # def compute_d_log_p(self, x_0, t_s):
    #     x_t, noise = self.apply_noise(x_0, t_s)
    #     d_log_p = -noise / self.sqrt_one_minus_alpha_cumprod
    #     return x_t, d_log_p


    def apply_noise(self, x_0, t_s):
        noise = torch.randn_like(x_0)

        mean_mult = get_items_reshaped(self.sqrt_alpha_cumprod, t_s, x_0.shape) 
        mean = mean_mult * x_0 
        
        variance_mult = get_items_reshaped(self.sqrt_one_minus_alpha_cumprod, t_s, x_0.shape)
        variance = variance_mult * noise
        
        x_t = mean + variance
        return x_t, noise


    def sample(self, shape, classes=None, w = 0.2):
        x_t = torch.randn(size=shape, device=self.device)

        if classes is None:
            classes = torch.zeros(size=[x_t.shape[0], ], device=x_t.device)

        with torch.no_grad():
            for t in reversed(range(1, self.T)):
                # make an array of t_s, one for each sample we want.
                t_s = torch.full(size=[shape[0]], fill_value=t, device=self.device)

                predicted_noise = self.forward(x_t, t_s, classes)

                # predicted_noise = (1 + w) * self.forward(x_t, t_s, classes) - \
                    # w * self.forward(x_t, t_s, torch.zeros_like(classes)) 

                mean = self.sqrt_alpha_inverse[t] * \
                (x_t - predicted_noise * self.betas[t] / self.sqrt_one_minus_alpha_cumprod[t]) 

                noise = torch.randn_like(x_t)
                x_t = mean + math.sqrt(self.betas[t]) * noise
        return x_t
