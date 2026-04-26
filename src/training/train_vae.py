import torch.optim as optim
from src.models.vae import HainesVAE, HainesVAE_Pro, elbo_loss, weighted_elbo_loss


def train_vae(train_loader, cfg):
    model = HainesVAE(latent_dim=cfg['latent_dim'])
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    losses = []

    model.train()
    print(f"Entraînement VAE ({cfg['epochs']} époques)...")

    for epoch in range(cfg['epochs']):
        total = 0
        for (x,) in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = elbo_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(train_loader.dataset)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Époque {epoch+1}/{cfg['epochs']} | Loss: {avg:.6f}")

    print("Entraînement VAE terminé.")
    return model, losses


def train_vae_pro(train_loader, cfg):
    model = HainesVAE_Pro(latent_dim=cfg['latent_dim'])
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    losses = []

    model.train()
    print(f"Entraînement VAE Pro — phase standard ({cfg['epochs']} époques)...")

    for epoch in range(cfg['epochs']):
        total = 0
        for (x,) in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = elbo_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(train_loader.dataset)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Époque {epoch+1}/{cfg['epochs']} | Loss: {avg:.6f}")

    print(f"Phase pondérée ({cfg['weighted_epochs']} époques)...")
    optimizer = optim.Adam(model.parameters(), lr=cfg.get('lr_weighted', 1e-4))

    for epoch in range(cfg['weighted_epochs']):
        total = 0
        for (x,) in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = weighted_elbo_loss(recon, x, mu, logvar, cfg.get('weight_scale', 5.0))
            loss.backward()
            optimizer.step()
            total += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Époque pondérée {epoch+1}/{cfg['weighted_epochs']} | Loss: {total/len(train_loader.dataset):.6f}")

    print("Entraînement VAE Pro terminé.")
    return model, losses
