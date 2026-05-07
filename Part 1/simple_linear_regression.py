# This code implements a simple linear regression model using gradient descent. The `SimpleLinearRegression` class has methods for fitting the model to the data and making predictions. The `fit` method updates the slope (m) and intercept (b) of the line based on the mean squared error cost function, while the `predict` method uses the learned parameters to make predictions on new data. The cost history is stored for analysis of the training process.
import numpy as np 

# Simple Linear Regression using Gradient Descent
class SimpleLinearRegression:
    # Initialize the model parameters
    def __init__(self, learning_rate=0.0000001, n_iters=1000):
        self.lr = learning_rate # learning rate for gradient descent
        self.n_iters = n_iters # number of iterations for training
        self.m = 0 # slope
        self.b = 0 # intercept
        self.cost_history = [] # to store cost at each iteration

    # Fit the model to the data
    def fit(self, X, y):
        n = len(X) # number of samples

        # Gradient Descent
        for _ in range(self.n_iters):
            y_pred = self.m * X + self.b # predicted values based on current parameters

             # Calculate gradients
            dm = (-2/n) * np.sum(X * (y - y_pred)) # gradient with respect to m
            db = (-2/n) * np.sum(y - y_pred) # gradient with respect to b

            # Update parameters
            self.m -= self.lr * dm # update slope
            self.b -= self.lr * db # update intercept

            # Calculate and store cost
            cost = (1/n) * np.sum((y - y_pred)**2) # mean squared error cost
            self.cost_history.append(cost) # store cost for analysis

    # Make predictions using the learned parameters
    def predict(self, X):
        return self.m * X + self.b