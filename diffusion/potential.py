import numpy as np
import torch 

class LinearInterpolation(torch.nn.Module):
    def __init__(self,prior, target, net):
        super().__init__()
        self.prior = prior
        self.target=target
        self.net = net

    def __call__(self, x, t):
        return self.energy(x, t)

    def probability(self, x, t):
        return  np.exp(-self.energy(x, t))

    def energy(self, x, t):
        return  (1 - t) * self.target.energy(x) + t * self.prior.energy(x) + t * (1-t) * (self.net(torch.hstack([x, t])))
        # return self.net(torch.hstack([x, t]))
