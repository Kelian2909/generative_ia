import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=10, cond_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, z, cond):
        return self.model(torch.cat([z, cond], dim=1))


class Discriminator(nn.Module):
    def __init__(self, cond_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1 + cond_dim, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 16), nn.LeakyReLU(0.2),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x, cond):
        return self.model(torch.cat([x, cond], dim=1))
