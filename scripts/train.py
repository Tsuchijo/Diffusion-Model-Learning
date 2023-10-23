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
flags.DEFINE_string("wandb_name", None, "wandb name.")
flags.mark_flags_as_required(["config"])

## Define the model we are training ##
#  2D checkerboard dataset 3d -> 2d (x,y, t) -> (x,y)
#  fully connected network MLP in between

class MLP_diffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 2),
        )
        self.double()
    
    def forward(self, x):
        return self.net(x)


def generate_normal_samples(N, input_dimension):
    # generate N samples from a 2D normal distribution
    mean = torch.zeros(input_dimension)
    cov = torch.eye(input_dimension)
    samples = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).sample((N,))
    return samples.double()

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

    ## define beta and alpha ##
    # beta is a linear function of the timestep 0 < beta_0 < beta_T < 1
    # is cumprod of 1-beta from 0 to T
    # T is total number of timesteps
    t_total = 1000
    betas = np.linspace(1e-4, 0.2, int(t_total))
    alphas = torch.cumprod(1 - torch.tensor(betas), 0)
    alpha_t = 1- torch.tensor(betas)

    ## Load the model ##
    model = MLP_diffusion().to('cuda')

    # randomize weights
    for p in model.parameters():
        torch.nn.init.normal_(p, 0, 0.05)

    ## Load the optimizer ##
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr)
    print("Starting training loop")
    ## Training loop ##
    x0 = next(inf_train_loader).numpy()
    plt.scatter(x0[:, 0], x0[:, 1])
    plt.show()
    # forward pass
    timestep = (torch.randint(0, t_total-1, (x0.shape[0], 1))).to('cuda')
    x0 = torch.from_numpy(x0)
    x_t, epsilon = forward_pass(x0.to('cpu'), timestep.to('cpu'), alphas)
    x_t = x_t.numpy()
    plt.scatter(x_t[:, 0], x_t[:, 1])
    plt.show()
    for iteration in range(config.training.n_iters):
        x0 = next(inf_train_loader).to('cuda')
        # forward pass
        # define timestep by random sampling from 1 - t_total
        timestep = (torch.randint(0, t_total-1, (x0.shape[0], 1))).to('cuda')
        x_t, epsilon = forward_pass(x0.to('cpu'), timestep.to('cpu'), alphas)
        # backward pass
        # append timestep to x_t
        x_t = torch.cat([x_t.to('cuda'), (timestep / float(t_total)).to('cuda')], axis=1)
        optimizer.zero_grad()
        eps_pred = model(x_t)
        loss = torch.nn.functional.mse_loss(eps_pred, epsilon.to('cuda'), reduction='none').mean()
        loss.backward()
        optimizer.step()
        if iteration % 500 == 0:
            print(loss.item())
        #wandb.log({"loss": loss.item()})
    
    ## Plot the results ##
    # run inference pass on test set
    model.eval().to('cuda')
    x0 = next(iter(val_loader))
    # x_val = inference_pass(x0, t_total, alpha_t, alphas, model)
    # detach and make numpy
    timestep = (torch.randint(0, t_total-1, (x0.shape[0], 1))).to('cuda')
    x_t, epsilon = forward_pass(x0.to('cpu'), timestep.to('cpu'), alphas)
    x_t = torch.cat([x_t.to('cuda'), (timestep / float(t_total)).to('cuda')], axis=1)
    optimizer.zero_grad()
    eps_pred = model(x_t).to('cpu')
    timestep = timestep.to('cpu')
    x_val = (x_t[:, :2].to('cpu') - eps_pred * np.sqrt(1 - alphas[timestep]))

    x_val = x_val.to('cpu').detach().numpy()
    plt.scatter(x_val[:, 0], x_val[:, 1])
    plt.show()

        

## Forward pass of the diffusion model using normal distribution of noise
# takes an input image then applies forward diffusion process to it
def forward_pass( x0, timesteps, alpha):
    # x0 is the input image, uses precalculated alpha_t from timeskipping
    # alpha_bar is the cumulative product of beta_t - 1 from 0 to t where t is the current timestep
    # calculate noise from normal distribution (epsilon)
    noise = generate_normal_samples(x0.shape[0], x0.shape[1])
    a_t = alpha[timesteps]
    epsilon = torch.sqrt(1 - a_t) * noise
    x_t = torch.sqrt(a_t) * x0 + epsilon
    return x_t, noise

## Inference pass of the diffusion model
# for a set number of timesteps, take in noise and then repeatedly apply the reverse diffusion process
# to get the original distribution
def inference_pass(x0, timesteps, alpha, alpha_bar, model):
    x_T = generate_normal_samples(x0.shape[0], x0.shape[1]).to('cuda')
    # reverse diffusion process in noise and then apply model repeatedly
    for i in reversed(range(timesteps)):
        a_t = alpha[i]
        a_bar_t = alpha_bar[i]
        timestep_tensor = torch.ones((x0.shape[0], 1)) * i 
        eps = model(torch.cat([x_T, (timestep_tensor / float(timesteps)).to('cuda')], axis=1).to('cuda')).to('cuda')
        z_t = generate_normal_samples(x0.shape[0], x0.shape[1]).to('cuda')
        if i == 0:
            x_T = 1/np.sqrt(a_t) * (x_T - (((1 - a_t) / np.sqrt(1 - a_bar_t)) * eps))
        else:
            x_T = 1/np.sqrt(a_t) * (x_T - (((1 - a_t) / np.sqrt(1 - a_bar_t)) * eps)) + np.sqrt(1 - a_t) * z_t
        print(torch.max(x_T))
       
    return x_T



def main(argv):
    #logging.init(config=FLAGS.config.to_dict(), tags=FLAGS.tags, name=FLAGS.wandb_name)
    train(FLAGS.config)
    #wandb.log({})



if __name__ == "__main__":
    app.run(main)