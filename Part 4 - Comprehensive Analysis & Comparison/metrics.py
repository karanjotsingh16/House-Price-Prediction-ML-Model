import numpy as np

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))

def mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))

# SAFE MAPE (FIXED)
def mape(y, y_pred):
    epsilon = 1e-8
    return np.mean(np.abs((y - y_pred) / (y + epsilon))) * 100

def r2_score(y, y_pred):
    ss_total = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    return 1 - (ss_res / ss_total)

def adjusted_r2(y, y_pred, k):
    n = len(y)
    r2 = r2_score(y, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def bootstrap_ci(y, y_pred, metric_func, n_bootstrap=50):
    scores = []
    n = len(y)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        scores.append(metric_func(y[idx], y_pred[idx]))

    return np.percentile(scores, [2.5, 97.5])