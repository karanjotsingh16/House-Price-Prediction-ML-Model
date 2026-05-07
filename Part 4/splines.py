import numpy as np

def create_spline_features(X, knots):
    """
    Create piecewise linear spline features
    """
    X = X.reshape(-1, 1)
    features = [X]

    for knot in knots:
        features.append(np.maximum(0, X - knot))

    return np.hstack(features)