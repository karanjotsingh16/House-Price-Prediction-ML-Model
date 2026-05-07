import numpy as np

class PolynomialRegression:
    def __init__(self, lambda_=10):
        self.lambda_ = lambda_
        self.theta = None

    def fit(self, X, y):
        n, m = X.shape

        I = np.eye(m)
        I[0, 0] = 0  # don't regularize bias

        XtX = X.T @ X

        # Use pseudo-inverse for stability
        self.theta = np.linalg.pinv(XtX + self.lambda_ * I) @ X.T @ y

    def predict(self, X):
        return X @ self.theta