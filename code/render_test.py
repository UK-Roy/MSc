import safety_gymnasium
import torch
import numpy as np
from network import SafetyNetwork

# Create the safety-task environment
# env = safety_gymnasium.make("SafetyCarGoal1-v0", render_mode="human")
env = safety_gymnasium.make("SafetyCarGoal2Debug-v0", render_mode="human")
# Reset the environment
obs, info = env.reset(seed=1)

safety_net = SafetyNetwork(34, 64)
# Load the trained safety network
safety_net.load_state_dict(torch.load('MSc\code\safety_net.pth'))

while True:
    # Sample a random action
    act = env.action_space.sample()
    state = obs[40:72]
    lidar_action = np.concatenate((state, act))
    data = torch.tensor(lidar_action, dtype=torch.float32)
    safety_value = safety_net(data)
    print(f"Safety: {safety_value.round()}")
    # Step the environment: costs are returned
    nxt_obs, reward, cost, terminated, truncated, info = env.step(act)
    obs = nxt_obs
    if terminated or truncated:
        obs, info = env.reset()