import numpy as np
import safety_gymnasium

# Load the learned A and B matrices initially
A = np.load('A_matrix.npy')  # Initial A matrix
B = np.load('B_matrix.npy')  # Initial B matrix

# Parameters for learning
learning_rate = 0.001  # Learning rate for updating A and B

# Function to predict next LiDAR using A and B
def predict_lidar(L_current, action, A, B):
    return L_current @ A.T + action @ B.T

# Create the safety-task environment
env = safety_gymnasium.make("SafetyCarGoal2-v0", render_mode="human")

# Reset the environment to get the initial observation
obs, info = env.reset(seed=1)

episode_num = 10

cur_eps = 0

while cur_eps < episode_num:
    # Extract current LiDAR readings from the observation
    L_current = obs[40:72]  # Assuming 'lidar' key in the observation

    # Sample a random action
    action = env.action_space.sample()  # Example: [F_left, F_right]

    # Predict the next LiDAR readings using the current A and B
    # L_current  # Reshape for matrix multiplication
    # Find indices where the LiDAR values are 0
    zero_indices = np.where(L_current == 0)  # Use np.where to find indices of zeros
    action_new = np.array(action)        # Ensure the action has correct shape (1 x 2)
    
    L_next_predicted = predict_lidar(L_current, action_new, A, B)
    # L_next_predicted[zero_indices] = 0.0

    # Step the environment and observe the actual next LiDAR readings
    nxt_obs, reward, cost, terminated, truncated, info = env.step(action)
    L_next_actual = np.array([nxt_obs[40:72]])  # Actual next LiDAR readings
    
    error = L_next_actual - L_next_predicted
    print(error)
    
    
    # print("Updated A matrix:", A)
    # print("Updated B matrix:", B)
    
    # Update the observation for the next step
    obs = nxt_obs

    # If the episode ends, reset the environment
    if terminated or truncated:
        obs, info = env.reset()
        cur_eps += 1

# print("Learned A matrix:", A)
# print("Learned B matrix:", B)

# Step 6: Save the matrices for future use
# np.save('A_matrix.npy', A)
# np.save('B_matrix.npy', B)

# import numpy as np
# import safety_gymnasium

# # Load the learned A and B matrices
# A = np.load('A_matrix.npy')  # Load the pre-trained A matrix
# B = np.load('B_matrix.npy')  # Load the pre-trained B matrix

# # Parameters
# SAFETY_THRESHOLD = 0.2  # Define a threshold for determining if the LiDAR value is too close (unsafe)
# learning_rate = 0.001  # Learning rate for updating A and B if needed

# # Function to predict the next LiDAR readings
# def predict_lidar(L_current, action, A, B):
#     return L_current @ A.T + action @ B.T

# # Function to optimize action if the predicted next state is unsafe
# def optimize_action(L_current, action, A, B, safety_threshold=SAFETY_THRESHOLD):
#     # Predict the next LiDAR readings using the current action
#     L_next_predicted = predict_lidar(L_current, action, A, B)

#     # Check for unsafe values in the predicted next LiDAR readings
#     if np.any(L_next_predicted < safety_threshold):
#         print("Predicted next state is UNSAFE. Optimizing action...")
#         # Adjust the action to try to stay in the safe zone
#         # Here we are reversing the action as a basic adjustment, you can apply more sophisticated techniques
#         action = action * -0.5  # Reverse and reduce the action to avoid unsafe state
#     else:
#         print("Predicted next state is SAFE.")
    
#     return action

# # Create the safety-task environment
# env = safety_gymnasium.make("SafetyCarGoal2Debug-v0", render_mode="human")

# # Reset the environment
# obs, info = env.reset(seed=1)

# while True:
#     # Extract current LiDAR readings from the observation
#     L_current = np.array([obs['lidar']])  # Assuming 'lidar' key in the observation

#     # Sample an initial random action (forces applied to the left and right wheels)
#     action = env.action_space.sample()  # Example: [F_left, F_right]
#     action = np.array([action])  # Reshape for matrix operations

#     # Optimize the action if the predicted next LiDAR readings are unsafe
#     optimized_action = optimize_action(L_current, action, A, B)

#     # Step the environment with the (possibly optimized) action
#     nxt_obs, reward, cost, terminated, truncated, info = env.step(optimized_action[0])

#     # Update the observation for the next loop
#     obs = nxt_obs

#     # Reset the environment if terminated or truncated
#     if terminated or truncated:
#         obs, info = env.reset()
