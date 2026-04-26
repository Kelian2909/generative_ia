import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

MAIN_COLOR = '#2563eb'
DARK_GREY = '#111317'
LIGHT_BG = '#f6f7f9'
RED = '#ef4444'
LIGHT_BLUE = '#60a5fa'

plt.rcParams['font.family'] = ['Inter', 'sans-serif']


def plot_eda(df):
    fig = plt.figure(figsize=(18, 10), facecolor=LIGHT_BG)

    ax1 = plt.subplot(2, 2, 1)
    sns.histplot(df['haines_index'], kde=True, color=MAIN_COLOR, ax=ax1)
    ax1.set_title("Distribution Continue de l'Indice", fontweight='bold')

    ax2 = plt.subplot(2, 2, 2)
    sns.boxplot(x=df['haines_index'], color=MAIN_COLOR, ax=ax2)
    ax2.set_title('Identification des Extrêmes (Outliers)', fontweight='bold')

    ax3 = plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='risk_level', palette=[MAIN_COLOR, DARK_GREY, '#94a3b8'], ax=ax3)
    ax3.set_title('Répartition des Niveaux de Risque', fontweight='bold')

    ax4 = plt.subplot(2, 2, 4)
    sns.scatterplot(
        data=df.sample(min(2000, len(df))),
        x='latitude', y='haines_index',
        alpha=0.3, color=MAIN_COLOR, ax=ax4
    )
    ax4.set_title('Relation Indice vs Latitude', fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_preprocessing(df):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), facecolor=LIGHT_BG)

    sns.histplot(df['haines_index'], kde=True, color=MAIN_COLOR, bins=20, ax=ax[0])
    ax[0].set_title("Distribution de l'Indice de Haines", fontweight='bold', color=DARK_GREY)

    stats.probplot(df['haines_index'].dropna(), dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set(color=DARK_GREY, markersize=4)
    ax[1].get_lines()[1].set_color(MAIN_COLOR)
    ax[1].set_title("QQ-Plot vs Loi Normale", fontweight='bold', color=DARK_GREY)

    plt.tight_layout()
    plt.show()


def plot_vae_convergence(losses):
    plt.figure(figsize=(10, 5), facecolor=LIGHT_BG)
    plt.plot(losses, color=MAIN_COLOR, lw=2)
    plt.title("Convergence de l'entraînement du VAE", fontweight='bold')
    plt.xlabel("Époques")
    plt.ylabel("Loss (ELBO)")
    plt.grid(alpha=0.3)
    plt.show()


def plot_kde(df, synth, label='Synthétique', title='Comparaison des Distributions', trigger=None):
    max_reel = df['haines_index'].max()
    max_synth = synth.flatten().max()

    plt.figure(figsize=(12, 6), facecolor=LIGHT_BG)
    sns.kdeplot(df['haines_index'], label='Réalité (ERA5)', color=DARK_GREY, lw=2, fill=True, alpha=0.1)
    sns.kdeplot(synth.flatten(), label=label, color=MAIN_COLOR, lw=2, linestyle='--')
    plt.axvline(max_reel, color=DARK_GREY, linestyle=':', alpha=0.5, label=f'Max Réel ({max_reel:.2f})')
    plt.axvline(max_synth, color=MAIN_COLOR, linestyle=':', lw=1.5, label=f'Max {label} ({max_synth:.2f})')
    if trigger is not None:
        plt.axvline(trigger, color=RED, linestyle='-', lw=1.5, label=f'Trigger ({trigger})')

    plt.title(title, fontweight='bold')
    plt.xlabel("Indice de Haines")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    print(f"Max Réel (ERA5)       : {max_reel:.2f}")
    print(f"Max Synthétique (IA)  : {max_synth:.2f}")
    print(f"VaR 99% Réelle        : {np.percentile(df['haines_index'], 99):.2f}")
    print(f"VaR 99% IA            : {np.percentile(synth, 99):.2f}")


def plot_models_comparison(df, haines_gan, h_synth_wgan):
    plt.figure(figsize=(14, 7), facecolor=LIGHT_BG)
    sns.set_style("whitegrid", {'axes.facecolor': LIGHT_BG})

    sns.kdeplot(df['haines_index'], label='Réalité (ERA5)', color=DARK_GREY, lw=3, fill=True, alpha=0.1)
    sns.kdeplot(haines_gan.flatten(), label='GAN Classique', color=LIGHT_BLUE, lw=2, linestyle='-.')
    sns.kdeplot(h_synth_wgan.flatten(), label='WGAN-GP (Tail Risk)', color=MAIN_COLOR, lw=3)
    plt.axvline(df['haines_index'].max(), color=RED, linestyle=':', label='Limite Historique')

    plt.title("Évolution des Modèles Génératifs", fontsize=16, fontweight='bold')
    plt.xlabel("Indice de Haines")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def plot_var_analysis(df, h_synth_wgan):
    var_99_reel = np.percentile(df['haines_index'], 99)
    var_99_wgan = np.percentile(h_synth_wgan, 99)

    plt.figure(figsize=(12, 6), facecolor=LIGHT_BG)
    sns.kdeplot(df['haines_index'], label='Réalité', color=DARK_GREY, lw=2, fill=True, alpha=0.1)
    sns.kdeplot(h_synth_wgan.flatten(), label='WGAN-GP', color=MAIN_COLOR, lw=3)
    plt.axvline(var_99_reel, color=DARK_GREY, linestyle='--', alpha=0.6)
    plt.axvline(var_99_wgan, color=MAIN_COLOR, linestyle='--', lw=2)
    plt.fill_betweenx([0, 0.08], var_99_wgan, h_synth_wgan.max(), color=RED, alpha=0.2)

    plt.title("Analyse du Risque Extrême (VaR 99%)", fontweight='bold')
    plt.xlabel("Indice de Haines")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def plot_qqplot(df, synth, label='Synthétique'):
    percs = np.linspace(0, 100, 200)
    q_real = np.percentile(df['haines_index'], percs)
    q_synth = np.percentile(synth.flatten(), percs)
    diag = [min(q_real.min(), q_synth.min()), max(q_real.max(), q_synth.max())]

    plt.figure(figsize=(7, 7), facecolor=LIGHT_BG)
    plt.scatter(q_real, q_synth, color=MAIN_COLOR, alpha=0.6, s=20)
    plt.plot(diag, diag, color=RED, lw=2, linestyle='--', label='Identité')

    plt.title(f"QQ-Plot : Réel vs {label}", fontweight='bold')
    plt.xlabel("Quantiles Réels (ERA5)")
    plt.ylabel(f"Quantiles Simulés ({label})")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_cgan_evaluation(real, gen, var_real, var_gen):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(real, label='Réalité', linewidth=2)
    sns.kdeplot(gen, label='cGAN', linestyle='--', linewidth=2)
    plt.title("Comparaison des distributions (KDE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(real, bins=50, alpha=0.5, label='Réalité')
    plt.hist(gen, bins=50, alpha=0.5, label='cGAN')
    plt.title("Histogrammes comparés")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.kdeplot(real, label='Réalité')
    sns.kdeplot(gen, label='cGAN')
    plt.axvline(var_real, linestyle='--', label='VaR réel')
    plt.axvline(var_gen, linestyle='--', label='VaR cGAN')
    plt.title("Analyse des extrêmes (queue de distribution)")
    plt.legend()
    plt.show()