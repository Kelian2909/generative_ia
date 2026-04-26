# Modélisation Générative du Risque Climatique

Ce projet applique des modèles d'IA générative pour simuler des scénarios atmosphériques synthétiques à partir de l'Indice de Haines, un indicateur continu du potentiel de feux extrêmes dérivé des données de réanalyse ERA5. L'objectif est d'enrichir les historiques observés avec des événements extrêmes plausibles pour la tarification actuarielle de produits d'assurance paramétriques.

## Fonctionnalités

Le pipeline couvre l'intégralité de la chaîne, des données climatiques brutes aux métriques assurantielles :

- **Chargement des données** : lecture des fichiers ERA5 au format NetCDF et calcul de l'Indice de Haines continu à partir des champs de température et d'humidité par niveau de pression
- **Prétraitement** : transformation de Box-Cox, normalisation MinMax et construction des DataLoaders PyTorch
- **Modélisation générative** : cinq variantes de modèles entraînées sur le même pipeline — VAE, VAE pondéré, GAN, WGAN-GP et GAN conditionnel
- **Génération conditionnelle** : le cGAN conditionne la sortie sur la latitude, la longitude et le mois pour produire des scénarios cohérents spatialement et saisonnièrement
- **Évaluation statistique** : distance de Wasserstein, test de Kolmogorov-Smirnov, VaR 99% et QQ-plots calculés de manière uniforme sur tous les modèles




## Structure du projet

```
generative_ia/
├── src/
│   ├── models/          # architectures VAE, GAN, WGAN-GP, cGAN
│   ├── training/        # boucles d'entraînement par modèle
│   ├── data_loader.py   # chargement ERA5 et calcul de l'Indice de Haines
│   ├── preprocessing.py # Box-Cox, normalisation, DataLoader
│   ├── evaluation.py    # métriques d'évaluations
│   ├── visualization.py # plot
│   └── actuarial.py     # prime pure et probabilité de dépassement
├── data/
│   └── haines_data.nc   # données atmosphériques ERA5
├── notebooks.ipynb      # pipeline complet avec visualisations
├── config.py            # hyperparamètres et seed
└── requirements.txt
```


## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproductibilité

Tous les résultats sont reproductibles grâce à une seed fixée en début de notebook :

```python
import config
config.set_seed()  # SEED = 100
```