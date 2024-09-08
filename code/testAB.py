import safety_gymnasium
import numpy as np

# Load the learned A and B matrices
A = np.load('A_matrix.npy')  # Load your trained A matrix
B = np.load('B_matrix.npy')  # Load your trained B matrix

# Safety threshold: if any LiDAR reading is below this threshold, the state is unsafe
SAFETY_THRESHOLD = 0.2

# Create the safety-task environment
env = safety_gymnasium.make("SafetyCarGoal2-v0", render_mode="human")

# Reset the environment to get the initial observation
obs, info = env.reset(seed=1)

while True:
    # Extract current LiDAR readings from the observation
    lidar_current = obs[40:72]  # Assuming 'lidar' key in the observation

    # Sample a random action
    action = env.action_space.sample()  # Example: [F_left, F_right]

    # Predict the next LiDAR readings using the learned A and B matrices
    L_current_new = np.array([lidar_current])  # Convert to 2D array for matrix multiplication
    action_new = np.array([action])  # Ensure the action is in the correct shape (1 x 2)
    
    # Predict the next LiDAR readings
    L_next_predicted = L_current_new @ A.T + action_new @ B.T

    # Step the environment using the current action
    nxt_obs, reward, cost, terminated, truncated, info = env.step(action)

    # Check if the predicted state is unsafe
    unsafe = np.any(L_next_predicted < SAFETY_THRESHOLD)

    if unsafe:
        print("Warning: Predicted next state is UNSAFE! Adjusting action...")

        # Adjust action to make it safer (for simplicity, here we stop the robot)
        action = np.array([0.0, 0.0])  # Stop the robot if the next state is unsafe
        L_next_predicted = L_current_new @ A.T + action @ B.T
        print("Adjusted next LiDAR readings:", L_next_predicted)

    else:
        print("Predicted next state is SAFE.")

    # Update observation with the new one
    obs = nxt_obs

    # Reset the environment if terminated or truncated
    if terminated or truncated:
        obs, info = env.reset()
