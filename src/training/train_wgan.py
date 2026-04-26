import torch
from src.models.wgan import Generator, Critic, compute_gradient_penalty


def train_wgan(train_loader, cfg):
    latent_dim = cfg['latent_dim']
    gen = Generator(latent_dim)
    critic = Critic()

    opt_G = torch.optim.Adam(gen.parameters(), lr=cfg['lr_g'], betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(critic.parameters(), lr=cfg['lr_c'], betas=(0.5, 0.9))

    n_critic = cfg.get('n_critic', 10)
    gp_lambda = cfg.get('gp_lambda', 10)

    print(f"Entraînement WGAN-GP ({cfg['epochs']} époques)...")

    for epoch in range(cfg['epochs']):
        for i, (real,) in enumerate(train_loader):
            opt_C.zero_grad()
            z = torch.randn(real.size(0), latent_dim)
            fake = gen(z)

            gp = compute_gradient_penalty(critic, real, fake.detach())
            loss_C = -torch.mean(critic(real)) + torch.mean(critic(fake.detach())) + gp_lambda * gp
            loss_C.backward()
            opt_C.step()

            if i % n_critic == 0:
                opt_G.zero_grad()
                samples = gen(z)
                std_penalty = 1.0 / (torch.std(samples) + 1e-6)
                g_loss = -torch.mean(critic(samples)) + 0.1 * std_penalty
                g_loss.backward()
                opt_G.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Époque {epoch+1}/{cfg['epochs']} | D: {loss_C.item():.4f} | G: {g_loss.item():.4f} | Std: {torch.std(samples).item():.4f}")

    print("Entraînement WGAN-GP terminé.")
    return gen, critic
