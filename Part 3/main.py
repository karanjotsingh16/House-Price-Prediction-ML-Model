from data import generate_data
from utils import fill_missing_mean, remove_outliers, standardize
from polynomial_features import create_polynomial_features
from polynomial_regression import PolynomialRegression
from metrics import mse, aic, bic

from splines import create_spline_features
from kernel_regression import KernelRegression
from bias_variance import bias_variance_analysis

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load + Clean
# -----------------------------
df = generate_data()
df = fill_missing_mean(df)
df = remove_outliers(df)

X = df.drop(columns=["price"]).values
y = df["price"].values

# -----------------------------
# Shuffle + Split
# -----------------------------
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

train_errors = []
val_errors = []

degrees = [1, 2, 3]

# -----------------------------
# Validation Curve
# -----------------------------
for degree in degrees:
    X_poly_train = create_polynomial_features(X_train, degree)
    X_poly_val = create_polynomial_features(X_val, degree)

    X_poly_train, mean, std = standardize(X_poly_train)
    X_poly_val = (X_poly_val - mean) / std

    model = PolynomialRegression(lambda_=0.1)
    model.fit(X_poly_train, y_train)

    train_errors.append(mse(y_train, model.predict(X_poly_train)))
    val_errors.append(mse(y_val, model.predict(X_poly_val)))

# Plot validation curve
plt.plot(degrees, train_errors, label="Train")
plt.plot(degrees, val_errors, label="Validation")
plt.xlabel("Degree")
plt.ylabel("Error")
plt.title("Validation Curve")
plt.legend()
plt.show()

# -----------------------------
# Learning Curve
# -----------------------------
sizes = np.linspace(200, len(X_train), 5, dtype=int)

train_curve = []
val_curve = []

best_degree = degrees[np.argmin(val_errors)]

for size in sizes:
    X_sub = X_train[:size]
    y_sub = y_train[:size]

    X_poly_sub = create_polynomial_features(X_sub, best_degree)
    X_poly_val = create_polynomial_features(X_val, best_degree)

    X_poly_sub, mean, std = standardize(X_poly_sub)
    X_poly_val = (X_poly_val - mean) / std

    model = PolynomialRegression()
    model.fit(X_poly_sub, y_sub)

    train_curve.append(mse(y_sub, model.predict(X_poly_sub)))
    val_curve.append(mse(y_val, model.predict(X_poly_val)))

plt.plot(sizes, train_curve, label="Train")
plt.plot(sizes, val_curve, label="Validation")
plt.title("Learning Curve")
plt.legend()
plt.show()

# -----------------------------
# Final Model + Metrics
# -----------------------------
X_poly = create_polynomial_features(X, best_degree)
X_poly, mean, std = standardize(X_poly)

model = PolynomialRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

print("Best Degree:", best_degree)
print("MSE:", mse(y, y_pred))
print("AIC:", aic(y, y_pred, len(model.theta)))
print("BIC:", bic(y, y_pred, len(model.theta)))




# -----------------------------
# 1. SPLINES
# -----------------------------

X_area = df['area'].values.reshape(-1, 1)

knots = np.percentile(X_area, [25, 50, 75])

X_spline = create_spline_features(X_area, knots)

X_spline, mean, std = standardize(X_spline)

model = PolynomialRegression(lr=1e-4)
model.fit(X_spline, y)

y_spline = model.predict(X_spline)

print("Spline model MSE:", np.mean((y - y_spline)**2))


# -----------------------------
# 2. KERNEL METHOD
# -----------------------------
kernel_model = KernelRegression(degree=2)
kernel_model.fit(X, y)

y_kernel = kernel_model.predict(X)

print("Kernel model MSE:", np.mean((y - y_kernel)**2))


# -----------------------------
# 3. BIAS-VARIANCE ANALYSIS
# -----------------------------
from polynomial_regression import PolynomialRegression

bias, variance = bias_variance_analysis(
    X_poly, y, PolynomialRegression, degree=best_degree
)

print("Bias:", bias)
print("Variance:", variance)