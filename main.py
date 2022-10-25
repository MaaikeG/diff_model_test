import torch
from torchvision import datasets, transforms
from diffusionmodel import DiffusionModel
from train import train
from plotting import plot_sample_noising
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.random.manual_seed(10)


train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: (x * 2) - 1)
                    ]),
    download = True,            
)

test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: (x * 2) - 1)
                    ])
)

from unet import UNet

net = UNet(in_channels=1, #B&W image
           out_channels=1, # outputs the estimated noise for each channel
           embed_dim=32,
           n_classes=10,
           attention_resolutions=[64],
           nonlinearity = torch.nn.SiLU)


device = 'cuda'
batch_size=64
T = 200

beta_schedule = torch.linspace(1e-4, 0.05, T)
diff_model = DiffusionModel(net=net, variance_schedule=beta_schedule, device=device, class_dropout=0.)
print(sum([len(p) for p in diff_model.parameters()]))

#plot_sample_noising(diff_model, train_data, n_timesteps = 20, n_samples=3)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
optimizer = Adam(diff_model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, cooldown=3, verbose=True)

def MSE_loss(prediction, ground_truth):
    loss = ((prediction - ground_truth)**2).mean()
    return loss

train(diff_model, 
      loss_fn=MSE_loss, 
      optimizer=optimizer, 
      data_loader=train_loader, 
      scheduler=scheduler, 
      n_iterations=100, 
      device=device, 
      plot_interval=1)
