import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.0000001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.m = 0
        self.b = 0
        self.cost_history = []

    def fit(self, X, y):
        n = len(X)

        for _ in range(self.n_iters):
            y_pred = self.m * X + self.b

            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            self.m -= self.lr * dm
            self.b -= self.lr * db

            cost = (1/n) * np.sum((y - y_pred)**2)
            self.cost_history.append(cost)

    def predict(self, X):
        return self.m * X + self.b