from data import generate_data
from utils import fill_missing_mean, fill_missing_median, remove_outliers
from multiple_linear_regression import SimpleLinearRegression

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate Data
# -----------------------------
df = generate_data()

print("Original Missing Values:\n", df.isnull().sum()) # check for missing values in the original dataset

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
df_mean = fill_missing_mean(df)
df_median = fill_missing_median(df)

print("\nAfter Mean Imputation:\n", df_mean.isnull().sum())
print("\nAfter Median Imputation:\n", df_median.isnull().sum())

# -----------------------------
# 3. Remove Outliers
# -----------------------------
df_clean = remove_outliers(df_mean)

print("\nOriginal shape:", df.shape)
print("After removing outliers:", df_clean.shape)

# -----------------------------
# 4. Prepare Data
# -----------------------------
X = df_clean[['area', 'bedrooms', 'age']].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = df_clean['price'].values

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Mean:", X.mean(axis=0))
print("Std:", X.std(axis=0))