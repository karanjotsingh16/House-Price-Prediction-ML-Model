from data import generate_data
from utils import fill_missing_mean, remove_outliers
from multiple_linear_regression import MultipleLinearRegression

import numpy as np

def feature_selection(model, feature_names):
    importance = np.abs(model.m)
    importance = importance / np.sum(importance)

    print("\nFeature Importance:")
    for name, val in zip(feature_names, importance):
        print(f"{name}: {val:.4f}")

    return importance
# -----------------------------
# 1. Load & Clean Data
# -----------------------------
df = generate_data()
df = fill_missing_mean(df)
df = remove_outliers(df)

X = df[['area', 'bedrooms', 'age']].values
y = df['price'].values

# Normalize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# -----------------------------
# 2. Train-Test Split
# -----------------------------
indices = np.arange(len(X))
np.random.shuffle(indices)

split = int(0.8 * len(X))
train_idx = indices[:split]
test_idx = indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# -----------------------------
# 3. Train Models
# -----------------------------
models = {
    "No Regularization": MultipleLinearRegression(),
    "Ridge (L2)": MultipleLinearRegression(reg_type='l2', lambda_=10),
    "Lasso (L1)": MultipleLinearRegression(reg_type='l1', lambda_=10)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = np.mean((y_test - y_pred)**2)

    ss_total = np.sum((y_test - np.mean(y_test))**2)
    ss_residual = np.sum((y_test - y_pred)**2)

    r2 = 1 - (ss_residual / ss_total)

    results[name] = {
        "model": model,
        "mse": mse,
        "r2": r2
    }

feature_names = ["area", "bedrooms", "age"]

importance = feature_selection(model, feature_names)

threshold = 0.1
selected_features = importance > threshold

print("\nSelected Features:", selected_features)

X_selected = X[:, selected_features]
model_selected = MultipleLinearRegression()
model_selected.fit(X_selected, y)
# -----------------------------
# 4. Print Results
# -----------------------------
print("\nModel Comparison:")
for name, res in results.items():
    print(f"\n{name}")
    print("MSE:", res["mse"])
    print("R2:", res["r2"])

# -----------------------------
# 5. Feature Importance
# -----------------------------
print("\nFeature Importance (Weights):")

for name, res in results.items():
    print(f"\n{name}:")
    print("Area:", res["model"].m[0])
    print("Bedrooms:", res["model"].m[1])
    print("Age:", res["model"].m[2])

# -----------------------------
# 6. Cross Validation (Simple)
# -----------------------------
lambdas = [0, 1, 10, 50]

print("\nCross Validation (Ridge):")

for lam in lambdas:
    model = MultipleLinearRegression(reg_type='l2', lambda_=lam)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = np.mean((y_test - y_pred)**2)
    print(f"Lambda {lam} → MSE: {mse}")

# -----------------------------
# 7. Predict Custom Input
# -----------------------------
new_data = np.array([[2500, 3, 10]])
new_data_norm = (new_data - mean) / std

best_model = results["Ridge (L2)"]["model"]
prediction = best_model.predict(new_data_norm)

print("\nPredicted Price:", prediction[0])