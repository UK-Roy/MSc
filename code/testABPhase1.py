import numpy as np
import safety_gymnasium
from scipy.optimize import minimize

# Load the learned A and B matrices
A = np.load('A_matrix.npy')  # Load the pre-trained A matrix
B = np.load('B_matrix.npy')  # Load the pre-trained B matrix

B_pseudo_inv = np.linalg.pinv(B)

# Parameters
SAFETY_THRESHOLD = 0.8  # Define a threshold for determining if the LiDAR value is too close (unsafe)
optimizer_steps = 50  # Number of steps for optimizing the action

# Function to predict the next LiDAR readings
def predict_lidar(L_current, action, A, B):
    return np.dot(L_current, A.T) + np.dot(action, B.T)

def del_lidar(L_current, L_next_predicted):
    return L_next_predicted - L_current

# Function to check whether the predicted LiDAR readings are safe
def unsafe(L_data, safety_threshold):
    return np.any(L_data >= safety_threshold)

def safe_action(L_current, L_next_predicted, action):
    if unsafe(L_data=L_next_predicted, safety_threshold=SAFETY_THRESHOLD):
        del_lidar_val = del_lidar(L_current=L_current, L_next_predicted=L_next_predicted)
        # Compute the pseudoinverse of A

        # Calculate Delta x using the pseudoinverse
        del_x = np.dot(B_pseudo_inv, del_lidar_val)
        action -= del_x
        # Predict the next LiDAR readings based on the current action
        L_next_predicted = predict_lidar(L_current, action, A, B)
    return action

# Create the safety-task environment
env = safety_gymnasium.make("SafetyCarGoal2-v0", render_mode="human")

# Reset the environment
obs, info = env.reset(seed=1)

while True:
    # Extract current LiDAR readings from the observation (assuming 'lidar' data starts from index 40)
    L_current = obs[40:72]  # Convert LiDAR readings to NumPy array

    # Initialize action with random values (forces on the wheels)
    action = env.action_space.sample()  # [F_left, F_right]

    # Predict the next LiDAR readings based on the current action
    L_next_predicted = predict_lidar(L_current, action, A, B)
    
    # Check if the state is unsafe
    if unsafe(L_next_predicted, SAFETY_THRESHOLD):
        print("Unsafe state detected, optimizing action...")
        optimized_action = safe_action(L_current=L_current, L_next_predicted=L_next_predicted, action=action)
        
    else:
        print("Safe state detected, using original action.")
        optimized_action = action

    # Step the environment with the optimized action
    nxt_obs, reward, cost, terminated, truncated, info = env.step(optimized_action)

    # Update the observation for the next loop
    obs = nxt_obs

    # Reset the environment if terminated or truncated
    if terminated or truncated:
        obs, info = env.reset()