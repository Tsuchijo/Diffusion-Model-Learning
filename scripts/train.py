import time

import matplotlib.pyplot as plt
import torch
from absl import app, flags
from common_utils import logging
from common_utils.random import RNG, set_random_seed
from ml_collections.config_flags import config_flags
from tqdm.auto import tqdm
import itertools
import numpy as np

import diffusion_models
from diffusion_models import data
import wandb
import gc

logging.support_unobserve()


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.")
flags.DEFINE_list("tags", [], "Tags to add to the run.")
from torch import nn

flags.DEFINE_string("wandb_name", None, "wandb name.")
flags.mark_flags_as_required(["config"])

## Define model ##
class fully_connected(nn.Module):
    hidden_dim = 128
    def  __init__(self):
        super(fully_connected, self).__init__()
        self.fc1 = nn.Linear(3, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 2)
        self.double()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) 
        x = torch.relu(self.fc3(x)) 
        x = (self.fc4(x))
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Define nosie schedule ##
t_total = 2000
beta = torch.linspace(1e-4, 0.02, t_total).to(device).unsqueeze(1)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


## Training loop ##
def train(config):
    if config.seed is not None:
        set_random_seed(config.seed)

    ## Load the dataset ##
    train_set, test_set = data.get_datasets(config.data)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.training.batch_size, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.training.batch_size, shuffle=False
    )
    inf_train_loader = itertools.cycle(train_loader)
    inf_val_loader = itertools.cycle(val_loader)

    model = fully_connected().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss().to(device)

    ## Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    x0 = next(inf_train_loader).to(device)
    print(device)
    for iteration in range(config.training.n_iters):
        x0 = next(inf_train_loader).to(device)
        x0 = normalize(x0)
        ## Create random timsteps
        timestep = (torch.randint(0, t_total-1, (x0.shape[0],)) ).to(device)
        x_t, eps = forward(x0, alpha_bar, timestep)

        ## Zero out gradients
        optimizer.zero_grad()
        eps_pred = model(torch.cat((x_t, (timestep.float() / t_total).unsqueeze(1)), dim=1))
        ## MSE loss between eps and eps predicted
        loss = loss_fn(eps_pred, eps)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ## Print loss
        if iteration % 100 == 0:
            print('Epoch: ', iteration, ' Loss: ',  loss.item(), end='\r')
    print('\n')
    ## Validation
    model.eval()
    x_pred, x_half, x_init = inference(x0, alpha_bar, alpha, t_total, model)

    ## Plot in matplotlib
    x_pred = x_pred.detach().cpu().numpy()
    x_half = x_half.detach().cpu().numpy()
    x_init = x_init.detach().cpu().numpy()
    x0 = x0.detach().cpu().numpy()
    plt.scatter(x0[:, 0], x0[:, 1])
    plt.scatter(x_pred[:, 0], x_pred[:, 1])
    #plt.scatter(x_half[:, 0], x_half[:, 1])
    plt.show()
    
## Forward pass ##
def forward(x0, alpha_bar, timestep):
    alphas_t = alpha_bar[timestep]
    noise = torch.randn_like(x0)
    x_t = torch.sqrt(alphas_t) * x0 + torch.sqrt(1.0 - alphas_t) * noise
    return x_t, noise

## Inference Pass ##
def inference(x0, alpha_bar, alpha, t_total, model):
    with torch.no_grad():
        x_T = torch.randn_like(x0)
        x_T_half = None
        x_T_start = x_T
        for t in reversed(range(t_total)):
            model.zero_grad()
            alpha_bar_t = alpha_bar[t]
            alpha_t = alpha[t]
            timestep = (torch.ones((x0.shape[0],)) * float(t) / t_total).to(device).unsqueeze(1).float()
            epsilon = model(torch.cat((x_T, timestep), dim=1))
            noise = torch.randn_like(x0).to(device)
            print('Iteration: ', t, end='\r')
            if t > 0:
                beta_bar_t = (1.0 - alpha_bar[t-1]) / (1.0 - alpha_bar_t) * (1 - alpha_t)
                x_T = (1.0 / torch.sqrt(alpha_t)) * (x_T - ((1 - alpha_t) / (1 - alpha_bar_t)) * epsilon) + torch.sqrt(beta_bar_t) * noise
            else:
                x_T = (1.0 / torch.sqrt(alpha_t)) * (x_T - ((1 - alpha_t) / (1 - alpha_bar_t)) * epsilon)

            if t == t_total // 2:
                x_T_half = x_T
        return x_T, x_T_half, x_T_start

## Normalize Sample sets variance and mean ##
def normalize(x):
    x = (x - x.mean(dim=0)) / x.std(dim=0)
    return x

def main(argv):
    train(FLAGS.config)



if __name__ == "__main__":
    app.run(main)