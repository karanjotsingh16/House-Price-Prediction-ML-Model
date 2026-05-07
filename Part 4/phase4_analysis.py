from data import generate_data
from utils import fill_missing_mean, remove_outliers, standardize
from polynomial_features import create_polynomial_features
from polynomial_regression import PolynomialRegression
from splines import create_spline_features
from kernel_regression import KernelRegression
from bias_variance import bias_variance_analysis
from metrics import *

import numpy as np

# -----------------------------
# Load & Clean Data
# -----------------------------
df = generate_data()
df = fill_missing_mean(df)
df = remove_outliers(df)

X = df.drop(columns=["price"]).values
y = df["price"].values

# -----------------------------
# Shuffle + Split
# -----------------------------
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 🔥 Polynomial Model (FIXED)
# -----------------------------
degree = 2

# STEP 1: SCALE ORIGINAL FEATURES
X_train_scaled, mean_x, std_x = standardize(X_train)
X_test_scaled = (X_test - mean_x) / std_x

# STEP 2: CREATE POLYNOMIAL FEATURES
X_poly_train = create_polynomial_features(X_train_scaled, degree)
X_poly_test = create_polynomial_features(X_test_scaled, degree)

# STEP 3: TRAIN
poly_model = PolynomialRegression(lambda_=10)
poly_model.fit(X_poly_train, y_train)

# STEP 4: PREDICT
y_poly = poly_model.predict(X_poly_test)

# -----------------------------
# Spline Model
# -----------------------------
X_area_train = X_train[:, 0].reshape(-1, 1)
X_area_test = X_test[:, 0].reshape(-1, 1)

knots = np.percentile(X_area_train, [25, 50, 75])

X_spline_train = create_spline_features(X_area_train, knots)
X_spline_test = create_spline_features(X_area_test, knots)

X_spline_train, mean_s, std_s = standardize(X_spline_train)
X_spline_test = (X_spline_test - mean_s) / std_s

spline_model = PolynomialRegression(lambda_=10)
spline_model.fit(X_spline_train, y_train)

y_spline = spline_model.predict(X_spline_test)

# -----------------------------
# Kernel Model (already correct)
# -----------------------------
X_train_k, mean_k, std_k = standardize(X_train)
X_test_k = (X_test - mean_k) / std_k

kernel_model = KernelRegression(degree=2, reg=1e-5)
kernel_model.fit(X_train_k, y_train)

y_kernel = kernel_model.predict(X_test_k)

# -----------------------------
# Metrics
# -----------------------------
models = {
    "Polynomial": y_poly,
    "Spline": y_spline,
    "Kernel": y_kernel
}

for name, pred in models.items():
    print(f"\n{name} Model")
    print("RMSE:", rmse(y_test, pred))
    print("MAE:", mae(y_test, pred))
    print("MAPE:", mape(y_test, pred))
    print("R2:", r2_score(y_test, pred))
    print("Adj R2:", adjusted_r2(y_test, pred, X_poly_train.shape[1]))

# -----------------------------
# Confidence Interval
# -----------------------------
ci = bootstrap_ci(y_test, y_poly, rmse)
print("\nPolynomial RMSE CI:", ci)

# -----------------------------
# Ensemble
# -----------------------------
y_ensemble = (y_poly + y_spline + y_kernel) / 3
print("\nEnsemble RMSE:", rmse(y_test, y_ensemble))

# -----------------------------
# Bias-Variance
# -----------------------------
bias, variance = bias_variance_analysis(
    X_poly_train, y_train, PolynomialRegression, degree
)

print("\nBias:", bias)
print("Variance:", variance)