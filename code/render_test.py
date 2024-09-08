import safety_gymnasium
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from network import LidarPredictionNetwork  # Assuming your model is in network.py
from sklearn.preprocessing import MinMaxScaler

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

# Load your trained model
model = LidarPredictionNetwork(input_size=2, output_size=32)
model_load_path = 'lidar_change_model_normalized.pth'
model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode

# Move model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the dataset from a CSV file
file_path = 'lidar_action_data.csv'
data = pd.read_csv(file_path)

# Assuming the first 32 columns are Δlidar_data and the last 2 columns are Δaction
lidar_changes = data.iloc[:, :32].values  # Δlidar_data
action_changes = data.iloc[:, 32:].values  # Δaction

# Normalization: Assuming you used normalization during training
# Create scalers for action and lidar if needed
action_scaler = MinMaxScaler(feature_range=(-1, 1))
lidar_scaler = MinMaxScaler(feature_range=(0, 1))  # Adjust based on how you scaled the lidar data

# Fit and transform both action and lidar data
lidar_changes_normalized = lidar_scaler.fit_transform(lidar_changes)
action_changes_normalized = action_scaler.fit_transform(action_changes)

# Create the safety-task environment
env = safety_gymnasium.make("SafetyCarGoal2-v0", render_mode="human")

# Reset the environment
obs, info = env.reset(seed=1)

# Initialize the previous LiDAR readings and actions
prev_lidar = np.array(obs[40:72])  # Assuming 'lidar' is part of the observation
prev_action = np.array([0.0, 0.0])  # Initialize action (e.g., no movement at the start)

# Metrics to accumulate the error
total_error = 0.0
num_steps = 0

while True:
    # Sample a random action (or use a specific policy)
    action = env.action_space.sample()
    
    # Step the environment
    nxt_obs, reward, cost, terminated, truncated, info = env.step(action)

    # Get the current LiDAR data and action
    current_lidar = np.array(nxt_obs[40:72])  # Assuming LiDAR data is in the observation

    # Calculate the actual Δlidar_data (difference between current and previous LiDAR)
    delta_lidar_actual = current_lidar - prev_lidar

    # Calculate the change in action (Δaction)
    delta_action = np.array(action) - prev_action

    # Apply normalization (if normalization was used during training)
    delta_action_normalized = action_scaler.transform([delta_action])  # Normalize action
    delta_action_tensor = torch.tensor(delta_action_normalized, dtype=torch.float32).to(device)

    # Predict the Δlidar_data using the trained model
    delta_lidar_predicted = model(delta_action_tensor).cpu().detach().numpy().squeeze()

    # If you scaled the LiDAR data during training, inverse transform the predicted values
    delta_lidar_predicted = lidar_scaler.inverse_transform([delta_lidar_predicted]).squeeze()

    # Calculate error (e.g., Mean Squared Error between actual and predicted Δlidar_data)
    error = np.mean((delta_lidar_predicted - delta_lidar_actual) ** 2)
    total_error += error
    num_steps += 1

    # Print the actual vs predicted comparison for inspection
    print(f"Step {num_steps}:")
    print(f"Actual Δlidar_data: {delta_lidar_actual}")
    print(f"Predicted Δlidar_data: {delta_lidar_predicted}")
    print(f"Step Error (MSE): {error:.6f}")
    
    # Update previous LiDAR and action for the next step
    prev_lidar = current_lidar
    prev_action = action

    # If episode is done, reset the environment
    if terminated or truncated:
        obs, info = env.reset()
        prev_lidar = np.array(obs[40:72])
        prev_action = np.array([0.0, 0.0])

    # If you want to exit after a certain number of steps for evaluation
    if num_steps >= 1000:  # Evaluate for 1000 steps
        break

# Calculate and print average error over all steps
avg_error = total_error / num_steps
print(f"Average MSE over {num_steps} steps: {avg_error:.6f}")
