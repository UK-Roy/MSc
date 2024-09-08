import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Load the dataset from a CSV file
file_path = 'lidar_action_data.csv'
data = pd.read_csv(file_path)

# Assuming the first 32 columns are Δlidar_data and the last 2 columns are Δaction
lidar_changes = data.iloc[:, :32].values  # Δlidar_data
action_changes = data.iloc[:, 32:].values  # Δaction

# Apply MinMaxScaler to normalize both lidar and action data
lidar_scaler = MinMaxScaler(feature_range=(0, 1))
action_scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit and transform both action and lidar data
lidar_changes_normalized = lidar_scaler.fit_transform(lidar_changes)
action_changes_normalized = action_scaler.fit_transform(action_changes)

# Convert the normalized data to PyTorch tensors
lidar_tensor = torch.tensor(lidar_changes_normalized, dtype=torch.float32)
action_tensor = torch.tensor(action_changes_normalized, dtype=torch.float32)

# Combine the input (Δaction) and output (Δlidar_data) into a dataset
dataset = TensorDataset(action_tensor, lidar_tensor)

# Split dataset into training (70%), validation (15%), and test (15%)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class LidarPredictionNetwork(nn.Module):
    def __init__(self, input_size=2, output_size=32, hidden_size=64):
        super(LidarPredictionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4, 16)
        self.fc3 = nn.Linear(16, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model and move it to the GPU if available
model = LidarPredictionNetwork(input_size=2, output_size=32).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    # Training phase
    for batch_action, batch_lidar in train_loader:
        # Move data to CUDA device
        batch_action, batch_lidar = batch_action.to(device), batch_lidar.to(device)
        
        optimizer.zero_grad()  # Zero the parameter gradients
        
        # Forward pass
        outputs = model(batch_action)
        
        # Compute the loss
        loss = criterion(outputs, batch_lidar)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
    
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for val_action, val_lidar in val_loader:
            # Move validation data to CUDA device
            val_action, val_lidar = val_action.to(device), val_lidar.to(device)
            
            # Forward pass
            val_outputs = model(val_action)
            
            # Compute the validation loss
            v_loss = criterion(val_outputs, val_lidar)
            val_loss += v_loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')

# Save the trained model
model_save_path = 'lidar_change_model_normalized.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the saved model
model = LidarPredictionNetwork(input_size=2, output_size=32).to(device)
model.load_state_dict(torch.load('lidar_change_model_normalized.pth'))
model.eval()  # Set to evaluation mode

# Test the model on the test dataset
test_loss = 0.0
with torch.no_grad():
    for test_action, test_lidar in test_loader:
        # Move test data to CUDA device
        test_action, test_lidar = test_action.to(device), test_lidar.to(device)
        
        # Forward pass
        test_outputs = model(test_action)
        
        # Compute the test loss
        t_loss = criterion(test_outputs, test_lidar)
        test_loss += t_loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f'Test Loss: {avg_test_loss:.4f}')

# Example usage for inference
new_action_change = np.array([[0.1, -0.05]])  # Example change in action
new_action_change_normalized = action_scaler.transform(new_action_change)  # Normalize action change

# Convert to tensor and predict
new_action_change_tensor = torch.tensor(new_action_change_normalized, dtype=torch.float32).to(device)
predicted_lidar_change_normalized = model(new_action_change_tensor).cpu().detach().numpy()

# Inverse transform to get original scale of LiDAR change
predicted_lidar_change = lidar_scaler.inverse_transform(predicted_lidar_change_normalized)
print("Predicted change in LiDAR data (original scale):", predicted_lidar_change)