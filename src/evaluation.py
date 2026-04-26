import numpy as np
import torch
from scipy.special import inv_boxcox
from scipy.stats import ks_2samp, wasserstein_distance


def generate_vae_samples(model, latent_dim, scaler, lmbda, shift, n=10000):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim)
        scaled = model.decoder(z).numpy()
    transformed = scaler.inverse_transform(scaled)
    return inv_boxcox(transformed, lmbda) - shift


def generate_gan_samples(generator, latent_dim, scaler, lmbda, shift, n=10000):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim)
        scaled = generator(z).numpy()
    transformed = scaler.inverse_transform(scaled)
    return inv_boxcox(transformed, lmbda) - shift


def generate_cgan_samples(generator, latent_dim, X_cond, scaler, lmbda, shift, n=10000):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim)
        idx = torch.randint(0, X_cond.shape[0], (n,))
        cond = X_cond[idx]
        scaled = generator(z, cond).numpy()
    transformed = scaler.inverse_transform(scaled)
    return inv_boxcox(transformed, lmbda) - shift


def compute_metrics(real, generated):
    gen_flat = generated.flatten()
    wd = wasserstein_distance(real, gen_flat)
    ks_stat, ks_pvalue = ks_2samp(real, gen_flat)
    var_99_real = np.percentile(real, 99)
    var_99_gen = np.percentile(gen_flat, 99)

    metrics = {
        'wasserstein': wd,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'var_99_real': var_99_real,
        'var_99_gen': var_99_gen,
        'max_real': real.max(),
        'max_gen': gen_flat.max(),
    }

    print("----- MÉTRIQUES -----")
    print(f"Wasserstein distance : {wd:.4f}")
    print(f"KS statistic        : {ks_stat:.4f}")
    print(f"KS p-value          : {ks_pvalue:.4f}")
    print(f"VaR 99% réel        : {var_99_real:.4f}")
    print(f"VaR 99% généré      : {var_99_gen:.4f}")
    print(f"Max réel            : {real.max():.2f}")
    print(f"Max généré          : {gen_flat.max():.2f}")

    return metrics
