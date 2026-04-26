import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from src.models.cgan import Generator, Discriminator


def prepare_cgan_data(df, batch_size=32):
    df = df.copy()
    df['month'] = pd.to_datetime(df['valid_time']).dt.month
    features = ['latitude', 'longitude', 'month']

    scaler_cond = MinMaxScaler()
    X_cond_np = scaler_cond.fit_transform(df[features])
    X_cond = torch.FloatTensor(X_cond_np)
    y = torch.FloatTensor(df['haines_final'].values).view(-1, 1)

    loader = DataLoader(TensorDataset(X_cond, y), batch_size=batch_size, shuffle=True)
    print(f"DataLoader cGAN prêt : {len(y)} échantillons, cond_dim={X_cond.shape[1]}")
    return loader, X_cond, scaler_cond


def train_cgan(train_loader, X_cond, cfg):
    latent_dim = cfg['latent_dim']
    cond_dim = X_cond.shape[1]

    generator = Generator(latent_dim, cond_dim)
    discriminator = Discriminator(cond_dim)

    g_opt = optim.Adam(generator.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    print(f"Entraînement cGAN ({cfg['epochs']} époques)...")

    for epoch in range(cfg['epochs']):
        for cond, real in train_loader:
            bs = real.size(0)
            real_labels = torch.ones(bs, 1)
            fake_labels = torch.zeros(bs, 1)

            d_opt.zero_grad()
            d_loss = criterion(discriminator(real, cond), real_labels)
            z = torch.randn(bs, latent_dim)
            fake = generator(z, cond)
            d_loss += criterion(discriminator(fake.detach(), cond), fake_labels)
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()
            g_loss = criterion(discriminator(fake, cond), real_labels)
            g_loss.backward()
            g_opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Époque {epoch+1}/{cfg['epochs']} | D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

    print("Entraînement cGAN terminé.")
    return generator, discriminator
