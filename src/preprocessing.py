import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def _categorize(val):
    if val < 7:
        return 'Low'
    if val < 9:
        return 'Moderate'
    return 'High'


def build_dataframe(haines_index):
    df = haines_index.to_dataframe(name='haines_index').reset_index()
    df = df.dropna()
    df['risk_level'] = df['haines_index'].apply(_categorize)
    return df


def fit_transform(df):
    shift = abs(df['haines_index'].min()) + 1
    data_for_transform = df['haines_index'] + shift
    df['haines_transformed'], lmbda = stats.boxcox(data_for_transform)

    scaler = MinMaxScaler()
    df['haines_final'] = scaler.fit_transform(df[['haines_transformed']])

    print(f"Lambda de Box-Cox : {lmbda:.2f}")
    return df, lmbda, scaler, shift


def print_stats(df):
    data = df['haines_index'].dropna()
    skewness = stats.skew(data)
    kurt = stats.kurtosis(data)
    shapiro_test = stats.shapiro(data.sample(min(len(data), 5000)))

    print("Structure Statistique de l'Indice :")
    print(f"- Skewness (Asymétrie)     : {skewness:.3f}")
    print(f"- Kurtosis (Aplatissement) : {kurt:.3f}")
    print(f"- Shapiro-Wilk (p-value)   : {shapiro_test.pvalue:.5f}")


def make_loader(df, batch_size=32):
    X = torch.FloatTensor(df['haines_final'].values.reshape(-1, 1))
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
    print(f"DataLoader prêt : {len(X)} échantillons")
    return loader, X
