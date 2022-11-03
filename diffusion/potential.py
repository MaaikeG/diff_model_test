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
        return self.net(torch.hstack([x, t]))
