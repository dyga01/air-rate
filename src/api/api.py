"""
Airbnb Rating Predictor API

This module implements a REST API using FastAPI that serves predictions from a trained neural network model.
The API predicts Airbnb listing ratings based on various features of the listing such as number of accommodates,
bathrooms, bedrooms, etc.

The module loads a pre-trained PyTorch model and feature scaler on startup and exposes endpoints
for making predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the model architecture
class AirbnbRatingPredictor(nn.Module):
    """
    Neural network model for predicting Airbnb ratings.

    The model consists of three fully connected hidden layers with ReLU activations,
    dropout for regularization, and batch normalization.

    Attributes:
        model (nn.Sequential): Sequential container of layers that make up the neural network.

    Args:
        input_dim (int): Number of input features.
        hidden_size_1 (int): Number of neurons in the first hidden layer.
        hidden_size_2 (int): Number of neurons in the second hidden layer.
        hidden_size_3 (int): Number of neurons in the third hidden layer.
        dropout_rate (float): Probability of dropping a neuron during training.
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
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor containing the features.

        Returns:
            torch.Tensor: Output tensor containing the predicted ratings.
        """
        return self.model(x)

# Define the request model
class ListingFeatures(BaseModel):
    """
    Pydantic model for Airbnb listing features used for prediction.

    Attributes:
        accommodates (int): Number of people the listing can accommodate.
        bathrooms (float): Number of bathrooms in the listing.
        bedrooms (int): Number of bedrooms in the listing.
        beds (int): Number of beds in the listing.
        price (float): Price per night in dollars.
        amenities_length (int): Number of amenities provided by the listing.
    """
    accommodates: int
    bathrooms: float
    bedrooms: int
    beds: int
    price: float
    amenities_length: int

# Initialize FastAPI app
app = FastAPI(
    title="Airbnb Rating Predictor API",
    description="API to predict Airbnb listing ratings based on various features",
    version="1.0.0",
)

# Load model and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "src/airbnb_rating_predictor.pth")  # Adjusted path
SCALER_PATH = os.path.join(os.path.dirname(BASE_DIR), "src/feature_scaler.pkl")

# Initialize global variables
model = None
scaler = None
has_scaler = False

@app.on_event("startup")
async def load_model_and_scaler():
    """
    Loads the pre-trained model and feature scaler when the API starts up.

    This function is automatically called by FastAPI during application startup.
    It initializes the model with the same architecture as used during training,
    loads the trained weights, and attempts to load the feature scaler.

    If the scaler file cannot be found, it will use approximate normalization values.

    Globals:
        model: The loaded PyTorch model.
        scaler: The loaded feature scaler.
        has_scaler: Boolean indicating whether a scaler was successfully loaded.

    Raises:
        Exception: Prints an error message if loading fails but doesn't stop the startup.
    """
    global model, scaler, has_scaler

    try:
        # Initialize model with same architecture as training
        model = AirbnbRatingPredictor(
            input_dim=6,
            hidden_size_1=128,
            hidden_size_2=64,
            hidden_size_3=32,
            dropout_rate=0.2
        ).to(device)

        # Load the trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        # Load the saved scaler if available
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            has_scaler = True
        else:
            print("Warning: Scaler file not found. Will use approximate normalization.")

    except Exception as e:
        print(f"Error loading model or scaler: {e}")

@app.get("/")
async def root():
    """
    Root endpoint for the API.

    Returns:
        dict: A simple message directing users to the prediction endpoint.
    """
    return {"message": "Airbnb Rating Predictor API. Use /predict endpoint to get predictions."}

@app.post("/predict")
async def predict(features: ListingFeatures):
    """
    Predicts the Airbnb rating based on listing features.

    This endpoint takes listing features as input, normalizes them,
    runs them through the model, and returns a predicted rating along
    with contextual feedback.

    Args:
        features (ListingFeatures): A Pydantic model containing the listing features.

    Returns:
        dict: A dictionary containing:
            - predicted_rating (float): The predicted rating (0-5 scale)
            - feedback (str): Contextual feedback based on the rating

    Raises:
        HTTPException:
            - 503 status code if the model is not loaded
            - 500 status code if there's an error during prediction
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    try:
        # Create input feature DataFrame
        feature_names = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'price', 'amenities_length']
        input_features = pd.DataFrame(
            [[features.accommodates, features.bathrooms, features.bedrooms,
              features.beds, features.price, features.amenities_length]],
            columns=feature_names
        )

        # Normalize the features
        if has_scaler:
            input_features_normalized = scaler.transform(input_features)
        else:
            # Use approximate values as fallback
            feature_means = np.array([3.5, 1.5, 1.5, 2.0, 150.0, 15.0])
            feature_stds = np.array([2.0, 0.7, 0.8, 1.0, 100.0, 8.0])
            input_features_normalized = (input_features.values - feature_means) / feature_stds

        # Convert to tensor
        input_tensor = torch.tensor(input_features_normalized, dtype=torch.float32).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_rating = prediction.item()

            # Ensure rating is within bounds
            predicted_rating = max(0, min(5, predicted_rating))

        # Generate contextual feedback
        if predicted_rating >= 4.7:
            feedback = "Exceptional! This is likely to be a highly sought-after listing."
        elif predicted_rating >= 4.5:
            feedback = "Excellent! This listing should perform very well."
        elif predicted_rating >= 4.0:
            feedback = "Good. This is around the average rating for successful listings."
        elif predicted_rating >= 3.5:
            feedback = "Average. Consider improving some aspects to stand out more."
        else:
            feedback = "Below average. You might want to consider enhancing several aspects of your listing."

        return {
            "predicted_rating": round(predicted_rating, 2),
            "feedback": feedback
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
