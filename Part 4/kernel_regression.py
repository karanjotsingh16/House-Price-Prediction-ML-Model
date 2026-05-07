import numpy as np

class KernelRegression:
    def __init__(self, degree=2, reg=1e-5):
        self.degree = degree
        self.reg = reg
        self.X_train = None
        self.y_train = None
        self.alpha = None

    def polynomial_kernel(self, X, Z):
        return (1 + X @ Z.T) ** self.degree

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        # Compute kernel matrix
        K = self.polynomial_kernel(X, X)

        # Add regularization (important)
        n = K.shape[0]
        K += self.reg * np.eye(n)

        # Solve system
        self.alpha = np.linalg.solve(K, y)

    def predict(self, X):
        K = self.polynomial_kernel(X, self.X_train)
        return K @ self.alpha