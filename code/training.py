import gymnasium as gym
import gym_simplegrid
from QLagent import QLearningAgent

env = gym.make('SimpleGrid-8x8-v0', render_mode='human')
agent = QLearningAgent(state_size=env.observation_space, action_size=env.action_space)

# Training Loop

num_episodes = 1000
for episode in range(num_episodes):

    obs, info = env.reset()
    done = env.unwrapped.done
    total_reward = 0

    for t in range(100):
        action = agent.choose_action(obs)
        next_obs , reward, done, _, info = env.step(obs)
        agent.learn(obs, action, reward, next_obs)
        obs = next_obs
        total_reward += reward
    
        if done:
            break
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")
        # env.render()
    
env.close()