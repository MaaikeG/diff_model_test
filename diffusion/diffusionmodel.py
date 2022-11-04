from typing import Callable, Tuple
import torch 


def default_loss_weighting(*args):
    '''a weighting function to weight the loss for each time t. 
    This corresponds to \lambda(t) in eq. (7) of the score-SDE paper.'''
    return 1.


class TransitionKernel():
    '''The (gaussian) transition probability distribution p(x_t | x_0, t)'''

    def __init__(self, cumulative_beta: Callable):
        self.cumulative_beta = cumulative_beta

    def mean(self, x_0: torch.Tensor, t: torch.Tensor):
        '''Compute the mean of the transition kernel, i.e. <p(x_t|x_0)>'''
        return x_0 * torch.exp(-self.cumulative_beta(t))

    def std(self, t: torch.Tensor):
        '''Compute the standard deviation of the transition kernel, i.e. the std of p(x_t|x_0)'''
        return torch.sqrt(1 - torch.exp(-self.cumulative_beta(t))) + 1e-8


class SDE():
    ''''''
    def __init__(self, transition_kernel: TransitionKernel):
        self.mean = transition_kernel.mean
        self.std = transition_kernel.std


    def forward_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''Given x_0 and time t, generate x_t according to SDE defined by the transition kernel'''
        noise = torch.randn_like(x_0)
        x_t = self.mean(x_0, t) + self.std(t) * noise
        return x_t


    def energy(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''Unnormalized negative log probability of sample x_t given probability distribution at time t conditioned on x_0.
        This is -log(p) = -log(exp(-U)) = U, i.e. from the unnormalized probability we get the energy.'''
        return 0.5 * ((x_t - self.mean(x_0, t)) / self.std(t))**2 
        

    def force(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''derivative of energy of the sample x_t given probability distribution at time t conditioned on x_0.
        This is -d/dx -log(p) = -d/dx -log(exp(-U)) = -d/dx U = F, i.e. from the unnormalized probability we get the energy and from that the force.'''
        return -(x_t - self.mean(x_0, t)) / self.std(t) 



class DiffusionModel(torch.nn.Module):
    '''The model estimates a potential by force-matching. '''
    def __init__(self,
                 potential: torch.nn.Module,
                 sde: SDE,
                 device='cuda', 
                 *args):

        super().__init__()

        self.device = device
        self.potential=potential
        self.sde = sde

        self.to(device)


    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Returns the force and time derivative of the energy at position x and time t.
        i.e. the gradient of the energy with respect to x and t respectively.'''       
        x.requires_grad_(True)
        t.requires_grad_(True)

        energies_t = self.energy(x, t).sum()
        forces_t = torch.autograd.grad(energies_t, x, create_graph=True)[0]
        dt = torch.autograd.grad(energies_t, t, create_graph=True)[0]
        return -forces_t, dt


    def force(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''The force at position x at time t. Does not conserve gradient information'''
        x.requires_grad_(True)

        energies_t = self.energy(x, t).sum()
        forces_t = torch.autograd.grad(energies_t, x, create_graph=True)[0]
        return -forces_t.detach()


    def energy(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''The energy at position x at time t.'''
        return self.potential(x, t) 


    def loss(self, 
             x_0:torch.Tensor, 
             t:torch.FloatType,  
             loss_fn_dx:Callable, 
             loss_fn_dt:Callable = lambda _: 0., 
             weighting_fn:Callable=default_loss_weighting) -> torch.Tensor:
        t_s = torch.full_like(x_0, t)

        x_t = self.sde.forward_sample(x_0, t_s)

        force = self.sde.force(x_0, x_t, t_s)
        
        dx, dt = self(x_t, t_s)

        weight = weighting_fn(t)

        return weight * (loss_fn_dx(dx, force) + loss_fn_dt(dt))
