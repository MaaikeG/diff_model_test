from typing import Callable
import torch 


class ZeroAlignedModel(torch.nn.Module):
    '''Wrapper for a net that returns the potential with an constant offset which ensures
    U(0, t) = 0.'''
    def __init__(self, potential: Callable):
        super().__init__()
        self.potential = potential

    def __call__(self, x, t):
        return self.potential(torch.hstack([x, t])) - self.potential(torch.hstack([torch.zeros_like(x), t]))


class LinearInterpolation(torch.nn.Module):
    '''The potential is a linear interpolation between the prior, the target, and the interpolating model. '''
    def __init__(self, prior: Callable, target: Callable, model: torch.nn.Module):
        super().__init__()
        self.prior = prior
        self.target=target
        self.model = model

    def __call__(self, x, t):
        '''Return the value of the potential at position x and time t.'''
        return  (1 - t) * self.target.energy(x) \
                + t * self.prior.energy(x) \
                + (t * (1-t)) * self.model(x, t)
