# This code generates a synthetic dataset for a regression problem, where the target variable is the price of a house based on its area. The dataset includes some missing values and outliers to make it more realistic for testing data preprocessing and modeling techniques. 

# import necessary libraries
import numpy as np
import pandas as pd

# function to generate synthetic data
def generate_data():
    np.random.seed(42) # for reproducibility
    n_samples = 10000 # number of samples to generate

    # Generate feature
    area = np.random.normal(2000, 500, n_samples) # mean=2000 sqft, std=500 sqft

    # Generate price
    price = 150 * area + np.random.normal(0, 20000, n_samples) # price = 150 * area + noise

    # Create DataFrame from the generated data
    df = pd.DataFrame({
        "area": area,
        "price": price
    })

    # Add missing values (5%)
    df.loc[df.sample(frac=0.05).index, 'area'] = np.nan # randomly set 5% of 'area' values to NaN

    # Add outliers (2%)
    outliers = df.sample(frac=0.02).index # randomly select 2% of the data to be outliers
    df.loc[outliers, 'price'] *= 3 # make the price of outliers 3 times higher than normal

    return df

