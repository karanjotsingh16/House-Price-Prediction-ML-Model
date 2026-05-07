import numpy as np

def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def aic(y, y_pred, k):
    n = len(y)
    mse_val = mse(y, y_pred)
    return n * np.log(mse_val) + 2 * k

def bic(y, y_pred, k):
    n = len(y)
    mse_val = mse(y, y_pred)
    return n * np.log(mse_val) + k * np.log(n)