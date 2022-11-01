from ast import arg
from turtle import forward
import torch
import math
from numpy import ndarray




class DiffusionModel(torch.nn.Module):
    def __init__(self,
                 potential,
                 device='cuda', 
                 *args):

        super().__init__()

        self.device = device
        self.potential=potential
        self.to(device)


    # actually, maybe revert some changes to go back to the way it was: all you change is the potential function in the input, which is now an interpolation between prior, target and NN
    def forward(self, x, t, dt):

        # g(t) is the variance of the noise at time t
        g_t = self.variance_schedule(t)

        # get prediction of the score at t...
        u_t = self.potential.force(x, t)

        # and multiply with the squared variance to get the inverse
        dx =  g_t**2 * u_t

        return dx


    def variance_schedule(t):
        # just do something linear now. Fix later.
        return 1 - t                     


    def compute_dx(self, x_0, t, dt):
        # eq (11) of SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS
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
