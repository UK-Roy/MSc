import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Load the dataset
data = pd.read_csv('lidar_action_data_for_AnB.csv')

# Assume the following structure:
# - Columns `lidar_current_1`, ..., `lidar_current_n` correspond to L_current
# - Columns `F_left`, `F_right` correspond to the action inputs
# - Columns `lidar_next_1`, ..., `lidar_next_n` correspond to L_next

n_lidar = 32  # Example: number of LiDAR beams (replace with the actual number)
current_lidar_cols = [f'lidar_current_{i}' for i in range(n_lidar)]
next_lidar_cols = [f'lidar_next_{i}' for i in range(n_lidar)]
action_cols = ['F_left', 'F_right']

# current_lidar_cols = [f'lidar_current_{i}' for i in range(32)] 
# action_cols = ['F_left', 'F_right'] 
# next_lidar_cols = [f'lidar_next_{i}' for i in range(32)]

# Extract current LiDAR readings, actions, and next LiDAR readings
L_current = data[current_lidar_cols].values  # Current LiDAR readings
actions = data[action_cols].values           # Actions (F_left and F_right)
L_next = data[next_lidar_cols].values        # Next LiDAR readings

# Step 2: Concatenate L_current and actions to form the input matrix
X = np.hstack((L_current, actions))  # Input: [L_current, F_left, F_right]

# Step 3: The output matrix Y is L_next (the next LiDAR readings)
Y = L_next

# Step 4: Fit a linear regression model to learn A and B matrices
model = LinearRegression()
model.fit(X, Y)

# The learned model coefficients correspond to the combined [A, B] matrix
coefficients = model.coef_  # This is an n_lidar x (n_lidar + 2) matrix
intercept = model.intercept_

# Step 5: Separate the learned matrix into A and B
A = coefficients[:, :n_lidar]  # First n_lidar columns are A
B = coefficients[:, n_lidar:]  # Last 2 columns are B (for actions F_left, F_right)

print("Learned A matrix:", A)
print("Learned B matrix:", B)

# Step 6: Save the matrices for future use
np.save('A_matrix.npy', A)
np.save('B_matrix.npy', B)

# Optional: You can also save the intercept if necessary
np.save('intercept.npy', intercept)