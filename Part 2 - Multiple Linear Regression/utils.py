import pandas as pd # for data manipulation

# Mean Imputation for missing values
def fill_missing_mean(df):
    df_copy = df.copy() # create a copy of the DataFrame to avoid modifying the original
    df_copy['area'] = df_copy['area'].fillna(df_copy['area'].mean()) # fill missing values in 'area' column with the mean of that column
    return df_copy

# Median Imputation for missing values
def fill_missing_median(df):
    df_copy = df.copy() # create a copy of the DataFrame to avoid modifying the original
    df_copy['area'] = df_copy['area'].fillna(df_copy['area'].median()) # fill missing values in 'area' column with the median of that column    
    return df_copy

# Remove Outliers using IQR
def remove_outliers(df):
    df_copy = df.copy() # create a copy of the DataFrame to avoid modifying the original

    # Calculate Q1, Q3, and IQR for the 'price' column
    Q1 = df_copy['price'].quantile(0.25)
    Q3 = df_copy['price'].quantile(0.75)
    IQR = Q3 - Q1

    # Define lower and upper bounds for outliers
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_clean = df_copy[(df_copy['price'] >= lower) & (df_copy['price'] <= upper)] # filter out outliers
    return df_clean