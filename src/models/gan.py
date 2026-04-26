import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 16), nn.LeakyReLU(0.2),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
