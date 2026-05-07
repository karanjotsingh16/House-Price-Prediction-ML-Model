import numpy as np

class PolynomialRegression:
    def __init__(self, lr=1e-3, n_iters=1000, lambda_=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_ = lambda_
        self.theta = None
        self.cost_history = []

    def fit(self, X, y):
        n, m = X.shape
        self.theta = np.zeros(m)

        for i in range(self.n_iters):
            y_pred = X @ self.theta
            error = y_pred - y

            gradients = (2/n) * (X.T @ error + self.lambda_ * self.theta)

            self.theta -= self.lr * gradients

            cost = (1/n) * np.sum(error**2)
            self.cost_history.append(cost)

            # Early stopping
            if i > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < 1e-5:
                break

    def predict(self, X):
        return X @ self.theta