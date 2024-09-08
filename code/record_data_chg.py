import safety_gymnasium  # Assuming you're using a SafetyGym environment
import numpy as np
import pandas as pd

# Initialize environment
env = safety_gymnasium.make('SafetyCarGoal2-v0')

# Reset the environment to get the initial observation
obs, info = env.reset()

# Initialize lists to collect data
lidar_current_list = []
action_list = []
lidar_next_list = []

# Number of steps for which you want to collect data
num_steps = 5000

for step in range(num_steps):
    # Extract current LiDAR readings from the observation
    lidar_current = obs[40:72]
    
    # Sample an action (you can also apply a specific action instead of sampling randomly)
    action = env.action_space.sample()  # Example: [F_left, F_right]
    
    # Take a step in the environment by applying the action
    next_obs, reward, cost, terminated, truncated, info = env.step(action)
    
    # Extract next LiDAR readings from the next observation
    lidar_next = next_obs[40:72]
    
    # Append the data to the lists
    lidar_current_list.append(lidar_current)
    action_list.append(action)
    lidar_next_list.append(lidar_next)
    
    # If the episode is done, reset the environment
    if terminated or truncated:
        obs, info = env.reset()
    else:
        obs = next_obs

# Convert the collected data into numpy arrays
lidar_current_array = np.array(lidar_current_list)
action_array = np.array(action_list)
lidar_next_array = np.array(lidar_next_list)

# Optionally save the data to a CSV file for future use
data = np.hstack((lidar_current_array, action_array, lidar_next_array))
column_names = [f'lidar_current_{i}' for i in range(lidar_current_array.shape[1])] + \
               ['F_left', 'F_right'] + \
               [f'lidar_next_{i}' for i in range(lidar_next_array.shape[1])]

df = pd.DataFrame(data, columns=column_names)
df.to_csv('lidar_action_data_for_AnB.csv', index=False)

print(f"Collected data saved to 'lidar_action_data.csv' with {num_steps} steps.")
