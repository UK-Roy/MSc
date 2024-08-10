import gymnasium as gym
import gym_simplegrid

# Load the default 8x8 map
env = gym.make('SimpleGrid-8x8-v0', render_mode='human')

# Load the default 4x4 map
env = gym.make('SimpleGrid-4x4-v0', render_mode='human')

# Load a custom map
obstacle_map = [
        "10001000",
        "10010000",
        "00000001",
        "01000001",
    ]

env = gym.make(
    'SimpleGrid-v0', 
    obstacle_map=obstacle_map, 
    render_mode='human'
)

# Use the options dict in the reset method
# This initialises the agent in location (0,0) and the goal in location (7,7)
env = gym.make('SimpleGrid-8x8-v0', render_mode='human')
obs, info = env.reset(options={'start_loc':0, 'goal_loc':63})