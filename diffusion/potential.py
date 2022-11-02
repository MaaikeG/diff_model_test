import numpy as np
import torch 

class LinearInterpolation(torch.nn.Module):
    def __init__(self,prior, target, net):
        super().__init__()
        self.prior = prior
        self.target=target
        self.net = net


    def probability(self, x, t):
        return  np.exp(-self.energy(x, t))

    def energy(self, x, t):
        # Here we have some options...
        # 1. the standard interpolation
        return  (1 - t) * self.target.energy(x) + t * self.prior.energy(x) + t * (1-t) * (self.net(torch.hstack([x, t])))

        # 2. We substract for each (x, t) the value at (0, t) (i.e. the center, that we are diffusing towards!). Then the energy at the center is the zero-energy. 
     #   return  (1 - t) * self.target.energy(x) + t * self.prior.energy(x) + t * (1-t) * (self.net(x, t) - self.net(torch.zeros_like(x), t))
