import numpy as np

class MultipleLinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000, reg_type=None, lambda_=0):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.reg_type = reg_type
        self.lambda_ = lambda_
        self.m = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.m = np.zeros(n_features)

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.m) + self.b
            error = y - y_pred

            dm = (-2/n_samples) * np.dot(X.T, error)
            db = (-2/n_samples) * np.sum(error)

            # Regularization
            if self.reg_type == 'l2':
                dm += (2 * self.lambda_ / n_samples) * self.m

            elif self.reg_type == 'l1':
                dm += (self.lambda_ / n_samples) * np.sign(self.m)

            self.m -= self.lr * dm
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.m) + self.b