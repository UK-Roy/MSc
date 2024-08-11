from qnetwork import QLearningAgent
from grid import GridWorldEnv

env = GridWorldEnv()
agent = QLearningAgent(state_size=4, action_size=4)

num_episodes = 500

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    for t in range(100):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")
        env.render()

print("Training completed.")
