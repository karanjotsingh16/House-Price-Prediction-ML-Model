import numpy as np

def fill_missing_mean(df):
    return df.fillna(df.mean())

def remove_outliers(df):
    df = df.copy()

    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df['price'] >= lower) & (df['price'] <= upper)]

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std