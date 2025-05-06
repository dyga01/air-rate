"""
This script evaluates a trained machine learning model on a test dataset and calculates evaluation metrics.

Steps:
1. Sets the model to evaluation mode.
2. Iterates through the test dataset using a DataLoader.
3. Computes predictions and compares them to actual values using a loss function.
4. Calculates evaluation metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.
5. Prints the metrics and returns them as a dictionary.

Output:
- A dictionary containing the evaluation metrics (MSE, RMSE, R-squared).
"""

import numpy as np
import torch

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test dataset and calculate metrics.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device to run the evaluation on (CPU or GPU).

    Returns:
        dict: A dictionary containing evaluation metrics (MSE, RMSE, R-squared).
    """
    model.eval()
    with torch.no_grad():
        test_loss = 0
        predictions_list = []
        actuals_list = []

        for features, targets in test_loader:
            # Move tensors to the configured device
            features, targets = features.to(device), targets.to(device)

            predictions = model(features)
            # Store predictions and actuals for analysis
            predictions_list.extend(predictions.cpu().numpy().flatten())
            actuals_list.extend(targets.cpu().numpy().flatten())

            loss = criterion(predictions, targets)
            test_loss += loss.item()

        # Calculate metrics
        test_mse = test_loss / len(test_loader)
        test_rmse = np.sqrt(test_mse)

        # Calculate R-squared
        predictions_array = np.array(predictions_list)
        actuals_array = np.array(actuals_list)
        corr_matrix = np.corrcoef(predictions_array, actuals_array)
        r_squared = corr_matrix[0, 1]**2

        metrics = {
            "MSE": test_mse,
            "RMSE": test_rmse,
            "R-squared": r_squared
        }

        print(f"Test MSE: {test_mse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"R-squared: {r_squared:.4f}")

        return metrics
