import numpy as np
from itertools import combinations_with_replacement

def create_polynomial_features(X, degree):
    n_samples, n_features = X.shape
    features = [np.ones(n_samples)]

    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            if len(combo) > 2:   # LIMIT interaction size (important)
                continue

            feature = np.ones(n_samples)
            for i in combo:
                feature *= X[:, i]

            features.append(feature)

    return np.column_stack(features)