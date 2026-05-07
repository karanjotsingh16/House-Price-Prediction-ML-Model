import numpy as np

class KernelRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.X_train = None
        self.y_train = None

    def polynomial_kernel(self, X, Z):
        return (1 + X @ Z.T) ** self.degree

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        K = self.polynomial_kernel(X, self.X_train)
        return K @ self.y_train / K.sum(axis=1)