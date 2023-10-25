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
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = torch.nn.Linear(hidden_dim, output_dim)
        self.double()
    
    def forward(self, x, timestep):
        y = self.layer1(x)
        x = torch.nn.functional.silu(y)
        y = self.layer2(x)
        x = torch.nn.functional.silu(y)
        y = self.layer3(x)
        x = torch.nn.functional.silu(y)
        y = self.layer4(x)
        return y
    

class PositionalEncodingNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_timesteps):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        
        # Positional encoding initialization
        self.position_enc = torch.nn.Parameter((position_encoding_init(num_timesteps, hidden_dim)), requires_grad=False)
        
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = torch.nn.Linear(hidden_dim, output_dim)
        self.layer5 = torch.nn.Linear(output_dim, output_dim)
        self.double()
    
    def forward(self, x, timestep):
        # Positional encoding
        pos_enc = self.position_enc[timestep].squeeze()
        y = self.layer1(x)
        x = torch.nn.functional.silu(y)
        y = self.layer2(x + pos_enc)
        x = torch.nn.functional.silu(x + y)
        y = self.layer3(x  + pos_enc)
        x = torch.nn.functional.silu(x + y)
        y = self.layer4(x + pos_enc)
        y = self.layer5(y)
        return y
    

## From https://github.com/ThiagoLira/ToyDiffusion/blob/main/diffusion.py
def position_encoding_init(n_position, d_pos_vec):
    ''' 
    Init the sinusoid position encoding table 
    n_position in num_timesteps and d_pos_vec is the embedding dimension
    '''
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(1000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).to(torch.float64)


def generate_normal_samples(N, input_dimension):
    # generate N samples from a 2D normal distribution
    mean = torch.zeros(input_dimension)
    cov = torch.eye(input_dimension)
    samples = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).sample((N,))
    return samples.double()

def generate_cosine_schedule(time_steps, s=0.008):
    t = np.linspace(0, 1, time_steps)
    f = np.square(np.cos(((t + s) / (1 + s)) * 0.5 * np.pi))
    return np.clip(torch.from_numpy(f / f[0]), 0.000001, 0.999999)



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

    ## define beta and alpha ##
    # beta is a linear function of the timestep 0 < beta_0 < beta_T < 1
    # is cumprod of 1-beta from 0 to T
    # T is total number of timesteps
    t_total = 1000
    betas = np.linspace(1e-4, 0.02, int(t_total))
    alphas = 1- torch.tensor(betas)
    alpha_bar = torch.cumprod(alphas, dim=0)

    
    pos_enc = position_encoding_init(t_total, 2).to('cuda')

    ## Load the model ##
    diffuser = MLP_diffusion(
        input_dim=3,
        hidden_dim=100,
        output_dim=2,
        num_timesteps=t_total
    ).to('cuda')

    # randomize weights
    for p in diffuser.parameters():
        torch.nn.init.normal_(p, 0, 0.05)

    ## Load the optimizer ##
    optimizer = torch.optim.Adam(diffuser.parameters(), lr=config.optim.lr)
    print("Starting training loop")
    ## Training loop ##
    x0 = next(inf_train_loader).numpy()
    # forward pass
    timestep = (torch.ones((x0.shape[0], 1)) * (t_total - 1)).to('cuda').int()
    x0 = torch.from_numpy(x0)
    x_t, epsilon = forward_pass(x0.to('cpu'), timestep.to('cpu'), alpha_bar)
    x_t = x_t.numpy()
    # create 3 subplots
    fig, axs = plt.subplots(1, 3)
    # set fig size
    fig.set_size_inches(18.5, 10.5)
    # plot x0
    axs[0].scatter(x_t[:, 0], x_t[:, 1])

    timestep = (torch.ones((x0.shape[0], 1)) * (25)).to('cuda').int()
    x_t, epsilon = forward_pass(x0.to('cpu'), timestep.to('cpu'), alpha_bar)
    axs[1].scatter(x_t[:, 0], x_t[:, 1])

    timestep = (torch.ones((x0.shape[0], 1)) * (1)).to('cuda').int()
    x_t, epsilon = forward_pass(x0.to('cpu'), timestep.to('cpu'), alpha_bar)
    axs[2].scatter(x_t[:, 0], x_t[:, 1])
    plt.show()
    loss_list = []
    for iteration in range(config.training.n_iters):
        x0 = next(inf_train_loader).to('cuda')

        # forward pass
        # define timestep by random sampling from 1 - t_total
        timestep = (torch.randint(0, t_total-1, (x0.shape[0], 1))).to('cuda')
        x_t, epsilon = forward_pass(x0.to('cpu'), timestep.to('cpu'), alpha_bar)
        # backward pass
        # append timestep to x_t
        x_t = x_t.to('cuda')
        #x_t = torch.cat([x_t.to('cuda'), (timestep / float(t_total)).to('cuda')], axis=1)
        # use positional encoding instead of timestep
        eps_pred = diffuser(torch.cat((x_t, timestep / 1000), axis=1), timestep)
        loss = torch.nn.functional.mse_loss(eps_pred, epsilon.to('cuda'), reduction='mean')
        optimizer.zero_grad()
        # iterate over batch and backprop
        loss.backward()
        optimizer.step()
        if iteration % 10 == 0:
            print(loss.item())
            #wandb.log({"loss": loss.item()})
        loss_list.append(loss.item())
    
    ## Plot the results ##
    # run inference pass on test set
    # plot loss
    plt.plot(loss_list)
    plt.show()
    diffuser = diffuser.eval()
    x0 = next(inf_val_loader).numpy()
    x_val = inference_pass(x0, t_total, alphas, alpha_bar, pos_enc, diffuser)
    print(x_val)
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
    x_t = (torch.sqrt(a_t) * x0) + epsilon
    return x_t, noise

## Inference pass of the diffusion model
# for a set number of timesteps, take in noise and then repeatedly apply the reverse diffusion process
# to get the original distribution
def inference_pass(x0, timesteps, alpha, alpha_bar, pos_enc, model):
    x_T = generate_normal_samples(x0.shape[0], x0.shape[1]).to('cuda')
    # reverse diffusion process in noise and then apply model repeatedly
    for i in reversed(range(timesteps)):
        a_t = alpha[i]
        a_bar_t = alpha_bar[i]
        timestep_tensor = (torch.ones((x0.shape[0], 1)) * i).int().to('cuda')
        model.zero_grad()
        eps = model(torch.cat((x_T, timestep_tensor / 1000), axis=1), timestep_tensor).to('cuda')
        z_t = generate_normal_samples(x0.shape[0], x0.shape[1]).to('cuda')
        if i < 1:
            x_T = 1/np.sqrt(a_t) * (x_T - (((1 - a_t) / np.sqrt(1 - a_bar_t)) * eps))
        else:
            beta_bar = (1 - alpha_bar[i - 1]) / (1 - alpha_bar[i]) * (1 - alpha[i])
            x_T = 1/np.sqrt(a_t) * (x_T - (((1 - a_t) / np.sqrt(1 - a_bar_t)) * eps)) + np.sqrt(beta_bar) * z_t
       
    return x_T



def main(argv):
    #logging.init(config=FLAGS.config.to_dict(), tags=FLAGS.tags, name=FLAGS.wandb_name)
    train(FLAGS.config)
    #wandb.log({})



if __name__ == "__main__":
    app.run(main)