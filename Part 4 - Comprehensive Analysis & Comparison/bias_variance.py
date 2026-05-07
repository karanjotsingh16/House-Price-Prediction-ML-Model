import numpy as np

def bias_variance_analysis(X, y, model_class, degree, n_runs=5):
    predictions = []

    for _ in range(n_runs):
        indices = np.random.choice(len(X), len(X), replace=True)
        X_sample = X[indices]
        y_sample = y[indices]

        model = model_class()
        model.fit(X_sample, y_sample)

        pred = model.predict(X)
        predictions.append(pred)

    predictions = np.array(predictions)

    mean_pred = np.mean(predictions, axis=0)

    bias = np.mean((mean_pred - y) ** 2)
    variance = np.mean(np.var(predictions, axis=0))

    return bias, variance