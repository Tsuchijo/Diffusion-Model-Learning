from ml_collections import ConfigDict, config_dict
import torch


def add_training_configs(config):
    config.training = training = ConfigDict()
    training.batch_size = 4024
    training.n_iters = 5000
    training.save_interval = config_dict.placeholder(int) # If given, will save a save of the model every save_interval iterations
    training.log_interval = 1000
    training.eval_interval = 1000
    training.clip_grad_norm = config_dict.placeholder(float) # If given, will clip the gradient norm to this value
    training.init_checkpoint = config_dict.placeholder(str) # If given, will load the model from this checkpoint

    # loss
    config.loss = loss = ConfigDict()

    # optimization
    config.optim = optim = ConfigDict()
    optim.optimizer = "Adam"
    optim.lr = 1.e-4


def add_sampling_configs(config):
    config.sampling = sampling = ConfigDict()


def add_data_configs(config):
    config.data = data = ConfigDict()
    data.dataset = "checkerboard"
    data.train_set_size = 1000000


def add_model_configs(config):
    config.model = model = ConfigDict()


def get_config(config_names=["model", "data", "training", "sampling"]):
    config = ConfigDict()
    config.seed = config_dict.placeholder(int)
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    for name in config_names:
        globals()[f"add_{name}_configs"](config)

    return config