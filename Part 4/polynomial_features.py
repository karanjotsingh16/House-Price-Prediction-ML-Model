import numpy as np

def create_polynomial_features(X, degree):
    n_samples, n_features = X.shape

    # Start with bias
    features = [np.ones(n_samples)]

    # Add original features
    features.append(X)

    if degree >= 2:
        # Add squared terms ONLY (no cross explosion)
        features.append(X ** 2)

    return np.column_stack(features)