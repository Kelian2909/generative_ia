import torch
import torch.nn as nn
import torch.optim as optim
from src.models.gan import Generator, Discriminator


def train_gan(train_loader, cfg):
    latent_dim = cfg['latent_dim']
    generator = Generator(latent_dim)
    discriminator = Discriminator()

    g_opt = optim.Adam(generator.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    print(f"Entraînement GAN ({cfg['epochs']} époques)...")

    for epoch in range(cfg['epochs']):
        for (real,) in train_loader:
            bs = real.size(0)
            real_labels = torch.ones(bs, 1)
            fake_labels = torch.zeros(bs, 1)

            d_opt.zero_grad()
            d_loss = criterion(discriminator(real), real_labels)
            z = torch.randn(bs, latent_dim)
            fake = generator(z)
            d_loss += criterion(discriminator(fake.detach()), fake_labels)
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()
            g_loss = criterion(discriminator(fake), real_labels)
            g_loss.backward()
            g_opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Époque {epoch+1}/{cfg['epochs']} | D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

    print("Entraînement GAN terminé.")
    return generator, discriminator
