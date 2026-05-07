import numpy as np
import pandas as pd

def generate_data():
    np.random.seed(42)
    n_samples = 10000

    area = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.poisson(3, n_samples) + 1
    bathrooms = bedrooms * 0.8 + np.random.normal(0, 0.5, n_samples)
    age = np.random.exponential(15, n_samples)
    distance_city = np.random.gamma(2, 3, n_samples)
    crime_rate = np.random.exponential(5, n_samples)
    school_rating = np.random.beta(2, 1, n_samples) * 9 + 1
    garage = np.random.binomial(3, 0.6, n_samples)
    basement = area * 0.3 + np.random.normal(0, 200, n_samples)

    price = (
        150 * area +
        10000 * bedrooms +
        8000 * bathrooms -
        300 * age -
        2000 * distance_city -
        1000 * crime_rate +
        5000 * school_rating +
        3000 * garage +
        50 * basement +
        0.01 * area**2 -
        100 * age * distance_city +
        np.random.normal(0, 20000, n_samples)
    )

    df = pd.DataFrame({
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "distance_city": distance_city,
        "crime_rate": crime_rate,
        "school_rating": school_rating,
        "garage": garage,
        "basement": basement,
        "price": price
    })

    # Missing values
    for col in df.columns:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    # Outliers
    df.loc[df.sample(frac=0.02).index, 'price'] *= 3

    return df