import gymnasium as gym
import gym_simplegrid
from QLagent import QLearningAgent

env = gym.make('SimpleGrid-8x8-v0', render_mode='human')
# obs, info = env.reset()
# done = env.unwrapped.done
agent = QLearningAgent(state_size=env.observation_space, action_size=env.action_space)

num_episodes = 1000
for episode in range(num_episodes):
    print("New Episode Started")
    obs, info = env.reset()
    done = env.unwrapped.done
    total_reward = 0
    
    for t in range(500):
        action = agent.choose_action(obs)
        # action = env.action_space.sample()
        next_obs, reward, done, _, info = env.step(action)
        agent.learn(obs, action, reward, next_obs)
        obs = next_obs
        total_reward += reward

        if done:
            print("Complete on episode")
            break

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")
        # env.render()
            
env.close()