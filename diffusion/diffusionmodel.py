import torch 


class TransitionKernel():
    '''Computes the mean and variance of the (gaussian) transition probability distribution,
    i.e. given x_0, the mean and variance of x_t'''

    def __init__(self, cumulative_beta: callable):
        self.cumulative_beta = cumulative_beta

    def mean(self, x_0: torch.Tensor, t: torch.Tensor):
        '''Compute the mean of the transition kernel, i.e. given x_0, the mean of p(x_t|x_0)'''
        return x_0 * torch.exp(-self.cumulative_beta(t))

    def std(self, t: torch.Tensor):
        '''Compute the standard deviation of the transition kernel, i.e. given x_0, the std of p(x_t|x_0)'''
        return torch.sqrt(1 - torch.exp(-self.cumulative_beta(t))) + 1e-8


class SDE():
    def __init__(self, transition_kernel: TransitionKernel):
        self.mean = transition_kernel.mean
        self.std = transition_kernel.std


    def forward_sample(self, x_0: torch.Tensor, t: torch.Tensor):
        '''Given x_0 and time t, generate x_t according to SDE defined by the transition kernel'''
        noise = torch.randn_like(x_0)
        x_t = self.mean(x_0, t) + self.std(t) * noise
        return x_t


    def energy(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        '''Unnormalized negative log probability of sample x_t given probability distribution at time t conditioned on x_0.
        This is -log(p) = -log(exp(-U)) = U, i.e. from the unnormalized probability we get the energy.'''
        return 0.5 * ((x_t - self.mean(x_0, t)) / self.std(t))**2 
        

    def force(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        '''derivative of energy of the sample x_t given probability distribution at time t conditioned on x_0.
        This is -d/dx -log(p) = -d/dx -log(exp(-U)) = -d/dx U = F, i.e. from the unnormalized probability we get the energy and from that the force.'''
        return -(x_t - self.mean(x_0, t)) / self.std(t) 



class DiffusionModel(torch.nn.Module):
    '''The model estimates a potential by force-matching. The potential is an interpolation
    between the prior, the target, and the estimated potential which is parameterized by a neural network.
    The interpolation between the three is determined by the interpolation function. '''
    def __init__(self,
                 potential: callable,
                 device='cuda', 
                 *args):

        super().__init__()

        self.device = device
        self.potential=potential
 
        self.to(device)


    def __call__(self, x, t):
      #  return self.potential(torch.hstack([x,t]))
        '''The energy at position x at time t.'''
        return self.force(x, t)

    def energy(self, x, t):
        '''The energy at position x at time t.'''
        return self.potential.energy(x, t)


    def force(self, x, t):
        '''Negative gradient of the energies at position x and time t, with respect to the input data x.'''       
        x.requires_grad_(True)
        t.requires_grad_(True)

        energies_t = self.energy(x, t).sum()
        forces_t = torch.autograd.grad(energies_t, x, create_graph=True)[0]
        dt = torch.autograd.grad(energies_t, t, create_graph=True)[0]
        return -forces_t, dt