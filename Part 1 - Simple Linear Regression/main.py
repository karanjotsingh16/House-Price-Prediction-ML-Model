from data import generate_data
from utils import fill_missing_mean, fill_missing_median, remove_outliers
from simple_linear_regression import SimpleLinearRegression

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
X = df_clean['area'].values
y = df_clean['price'].values

# -----------------------------
# 5. Train Model
# -----------------------------
model = SimpleLinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("\nSlope (m):", model.m)
print("Intercept (b):", model.b)

# -----------------------------
# 6. Plot Regression Line
# -----------------------------
plt.scatter(X, y, alpha=0.3, label="Data")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# -----------------------------
# 7. Residual Plot
# -----------------------------
residuals = y - y_pred

plt.scatter(X, residuals, alpha=0.3)
plt.axhline(y=0, color='red')
plt.xlabel("Area")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# -----------------------------
# 8. Cost Convergence Plot
# -----------------------------
plt.plot(model.cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()

# -----------------------------
# 9. Confidence Interval
# -----------------------------
sigma = np.std(residuals)
z = 1.96  # 95% confidence

lower_bound = y_pred - z * sigma
upper_bound = y_pred + z * sigma

plt.scatter(X, y, alpha=0.3, label="Data")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.fill_between(X, lower_bound, upper_bound, color='gray', alpha=0.3, label="Confidence Interval")

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Regression with Confidence Interval")
plt.legend()
plt.show()