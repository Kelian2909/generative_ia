import random
import numpy as np
import torch

SEED = 100

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

DATA_PATH = 'data/haines_data.nc'
BATCH_SIZE = 32
N_SAMPLES = 10000
TRIGGER = 25.0
INDEMNITE = 10000

VAE = dict(latent_dim=2, lr=1e-3, epochs=10)

VAE_PRO = dict(
    latent_dim=2,
    lr=5e-4,
    epochs=50,
    lr_weighted=1e-4,
    weighted_epochs=50,
    weight_scale=5.0,
)

GAN = dict(latent_dim=10, lr=0.0002, epochs=50)

WGAN = dict(
    latent_dim=64,
    lr_g=0.00002,
    lr_c=0.0001,
    epochs=200,
    n_critic=10,
    gp_lambda=10,
)

CGAN = dict(
    latent_dim=10,
    lr=0.0002,
    epochs=200,
    cond_features=['latitude', 'longitude', 'month'],
)
