"""
This script predicts the Airbnb rating for a listing based on user-provided features.

Steps:
1. Defines the neural network architecture used for prediction.
2. Loads the trained model and feature scaler (if available).
3. Accepts user input for listing features such as number of bedrooms, bathrooms, etc.
4. Normalizes the input features using the saved scaler or approximate values.
5. Uses the trained model to predict the rating.
6. Outputs the predicted rating along with contextual feedback.

Requirements:
- A trained model saved as 'airbnb_rating_predictor.pth'.
- A feature scaler saved as 'feature_scaler.pkl' (optional).

Output:
- Predicted rating on a scale of 0 to 5.
- Contextual feedback based on the predicted rating.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os

# Define the model architecture
class AirbnbRatingPredictor(nn.Module):
    """
    Neural network model for predicting Airbnb ratings.

    Args:
        input_dim (int): Number of input features.
        hidden_size_1 (int): Number of neurons in the first hidden layer.
        hidden_size_2 (int): Number of neurons in the second hidden layer.
        hidden_size_3 (int): Number of neurons in the third hidden layer.
        dropout_rate (float): Dropout rate for regularization.
    """
    def __init__(self, input_dim, hidden_size_1, hidden_size_2, hidden_size_3, dropout_rate):
        super(AirbnbRatingPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size_2),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, 1)
        )

    def forward(self, x):
        return self.model(x)

def predict_rating():
    """
    Predicts the Airbnb rating based on user-provided features.

    Steps:
    1. Loads the trained model and scaler.
    2. Accepts user input for listing features.
    3. Normalizes the input features.
    4. Uses the model to predict the rating.
    5. Outputs the predicted rating and contextual feedback.
    """
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model and scaler paths
    model_path = "airbnb_rating_predictor.pth"
    scaler_path = "feature_scaler.pkl"

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return

    # Check if the scaler file exists
    if not os.path.exists(scaler_path):
        print(f"Warning: Scaler file '{scaler_path}' not found. Will use approximate normalization.")
        has_scaler = False
    else:
        # Load the saved scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        has_scaler = True

    # Prompt user for input features
    print("\n=== Airbnb Rating Predictor ===")
    print("Please enter the following details about your listing:")

    try:
        # Collect user input for features
        accommodates = float(input("Number of people it accommodates: "))
        bathrooms = float(input("Number of bathrooms: "))
        bedrooms = float(input("Number of bedrooms: "))
        beds = float(input("Number of beds: "))
        price = float(input("Price per night (in dollars, without $ sign): "))
        amenities_length = int(input("Number of amenities: "))

        # Create input feature DataFrame with proper column names
        feature_names = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'amenities_length']
        input_features = pd.DataFrame(
            [[accommodates, bathrooms, bedrooms, beds, price, amenities_length]],
            columns=feature_names
        )

        # Normalize the features using the saved scaler or approximate values
        if has_scaler:
            input_features_normalized = scaler.transform(input_features)
        else:
            # Use approximate values as fallback
            feature_means = np.array([3.5, 1.5, 1.5, 2.0, 150.0, 15.0])
            feature_stds = np.array([2.0, 0.7, 0.8, 1.0, 100.0, 8.0])
            input_features_normalized = (input_features.values - feature_means) / feature_stds

        # Convert to tensor
        input_tensor = torch.tensor(input_features_normalized, dtype=torch.float32).to(device)

        # Initialize model with same architecture as training
        model = AirbnbRatingPredictor(
            input_dim=6,
            hidden_size_1=128,
            hidden_size_2=64,
            hidden_size_3=32,
            dropout_rate=0.2
        ).to(device)

        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Set to evaluation mode
        model.eval()

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_rating = prediction.item()

            # Airbnb ratings are typically on a scale of 0-5
            predicted_rating = max(0, min(5, predicted_rating))

        # Output the predicted rating
        print(f"\nPredicted Rating: {predicted_rating:.2f} out of 5")

        # Provide contextual feedback
        if predicted_rating >= 4.7:
            print("Exceptional! This is likely to be a highly sought-after listing.")
        elif predicted_rating >= 4.5:
            print("Excellent! This listing should perform very well.")
        elif predicted_rating >= 4.0:
            print("Good. This is around the average rating for successful listings.")
        elif predicted_rating >= 3.5:
            print("Average. Consider improving some aspects to stand out more.")
        else:
            print("Below average. You might want to consider enhancing several aspects of your listing.")

    except ValueError:
        print("Error: Please enter numeric values where required.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    predict_rating()
