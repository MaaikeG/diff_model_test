from ast import arg
from turtle import forward
import torch
import math
from numpy import ndarray




def get_items_reshaped(arr, indices, shape):
    items = arr[indices]
    while len(shape) > items.ndim:
        items = items[..., None]
    items = torch.broadcast_to(items, shape)
    return items



class DiffusionModel(torch.nn.Module):
    def __init__(self,
                 potential,
                 device='cuda', 
                 *args):

        super().__init__()

        self.device = device
        self.potential=potential
        self.to(device)

        
    @property
    def T(self):
        return len(self.betas)


    def forward(self, x_t, t_s, classes):
        # perform dropout on the conditional information so we don't rely on only the class
        classes = self.condition_information_dropout(classes.float())
        return self.net(x_t, t_s, classes)


    def variance_schedule(t):
        # just do something linear now. Fix later.
        return 1 - t                     


    def apply_noise(self, x_0, t, dt):
        noise = torch.randn_like(x_0)
        beta = self.noise_schedule(t)
        dx = - 0.5 * beta * x_0 * dt + math.sqrt(beta) * noise
        return dx


    def sample(self, shape, classes=None, w = 0.2):
        x_t = torch.randn(size=shape, device=self.device)

        if classes is None:
            classes = torch.zeros(size=[x_t.shape[0], ], device=x_t.device)

        with torch.no_grad():
            for t in reversed(range(1, self.T)):
                # make an array of t_s, one for each sample we want.
                t_s = torch.full(size=[shape[0]], fill_value=t, device=self.device)

            #    predicted_noise = self.forward(x_t, t_s, classes)

                predicted_noise = (1 + w) * self.forward(x_t, t_s, classes) - \
                     w * self.forward(x_t, t_s, torch.zeros_like(classes)) 

                mean = self.sqrt_alpha_inverse[t] * \
                (x_t - predicted_noise * self.betas[t] / self.sqrt_one_minus_alpha_cumprod[t]) 

                noise = torch.randn_like(x_t)
                x_t = mean + math.sqrt(self.betas[t]) * noise
        return x_t
