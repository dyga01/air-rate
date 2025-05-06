"""
This script preprocesses the Airbnb dataset for training and prediction.

Steps:
1. Reads the input CSV file containing Airbnb listing data.
2. Cleans and formats the 'price' column by removing special characters and converting it to a numeric type.
3. Handles missing values in numerical columns using the median value.
4. Extracts relevant features and the target variable for training.
5. Standardizes the feature values using `StandardScaler` for better model performance.
6. Saves the fitted scaler as 'feature_scaler.pkl' for use during prediction.

Output:
- Preprocessed feature matrix (X) and target vector (y).
- Saved scaler object for consistent feature scaling during prediction.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import pickle

def preprocess_data(data_path):
    """
    Preprocesses the Airbnb dataset for training.

    Args:
        data_path (str): Path to the input CSV file.

    Returns:
        tuple: A tuple containing:
            - X_scaled (numpy.ndarray): Standardized feature matrix.
            - y (pandas.Series): Target variable (review scores rating).
    """
    # Read the CSV file
    df = pd.read_csv(data_path)

    # Handle price column (remove $ and convert to float)
    if 'price' in df.columns:
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

    # Handle missing values
    numerical_cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'amenities_length']
    imputer = SimpleImputer(strategy='median')  # Replace missing values with the median
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Create features and target
    X = df[['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'amenities_length']]
    y = df['review_scores_rating']

    # Standardize features
    scaler = StandardScaler()  # Standardize features to have mean 0 and variance 1
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for later use in predictions
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return X_scaled, y