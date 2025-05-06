"""
This script trains a machine learning model to predict Airbnb ratings based on preprocessed data.

Steps:
1. Preprocesses the data using the `preprocess_data` function.
2. Splits the data into training and testing sets.
3. Defines a PyTorch dataset and dataloader for efficient data handling.
4. Builds a neural network model for rating prediction.
5. Trains the model using Mean Squared Error (MSE) loss and Adam optimizer.
6. Implements a learning rate scheduler to adjust learning rate during training.
7. Saves the best model based on validation loss.
8. Evaluates the model on the test dataset using the `evaluate_model` function.

Output:
- Trained model saved as 'airbnb_rating_predictor.pth'.
- Evaluation metrics printed to the console.
"""

import torch
import torch.nn as nn
import os
from preprocess import preprocess_data
from evaluate import evaluate_model
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Hyperparameters
BATCH_SIZE = 32  # Number of samples per batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
EPOCHS = 30  # Number of training epochs
TEST_SIZE = 0.2  # Proportion of data to use for testing
RANDOM_SEED = 42  # Seed for reproducibility
DROPOUT_RATE = 0.2  # Dropout rate for regularization
HIDDEN_LAYER_1_SIZE = 128  # Number of neurons in the first hidden layer
HIDDEN_LAYER_2_SIZE = 64  # Number of neurons in the second hidden layer
HIDDEN_LAYER_3_SIZE = 32  # Number of neurons in the third hidden layer
MODEL_SAVE_PATH = "airbnb_rating_predictor.pth"  # Path to save the trained model

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess the data
data_path = os.path.dirname(os.path.abspath(__file__)) + "/data/filtered_output.csv"
X, y = preprocess_data(data_path)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Define a custom PyTorch dataset
class AirbnbDataset(Dataset):
    """
    Custom dataset for Airbnb data.

    Args:
        features (array-like): Feature matrix.
        targets (array-like): Target values.
    """
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create datasets and dataloaders
train_dataset = AirbnbDataset(X_train, y_train)
test_dataset = AirbnbDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the model for rating prediction
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

# Initialize the model
input_dim = X_train.shape[1]
model = AirbnbRatingPredictor(
    input_dim=input_dim,
    hidden_size_1=HIDDEN_LAYER_1_SIZE,
    hidden_size_2=HIDDEN_LAYER_2_SIZE,
    hidden_size_3=HIDDEN_LAYER_3_SIZE,
    dropout_rate=DROPOUT_RATE
).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,  # Reduce learning rate by half on plateau
    patience=3   # Wait 3 epochs with no improvement before reducing
)

# Train the model
best_val_loss = float('inf')  # Initialize best validation loss
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    epoch_loss = 0
    for features, targets in train_loader:
        # Move tensors to the configured device
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the gradients
        predictions = model(features)  # Forward pass
        loss = criterion(predictions, targets)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_loader)  # Average training loss

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            val_loss += loss.item()

    val_loss = val_loss / len(test_loader)  # Average validation loss

    # Update the scheduler
    scheduler.step(val_loss)

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # Track best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)  # Save the best model

    # Print progress
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

# Load the best model for evaluation
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Evaluate the model
metrics = evaluate_model(model, test_loader, criterion, device)

print("Model trained and saved successfully!")
print(f"Best validation loss: {best_val_loss:.4f}")
