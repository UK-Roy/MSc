import safety_gymnasium
import pandas as pd
import numpy as np

# Create the safety-task environment
# env = safety_gymnasium.make("SafetyCarGoal1-v0", render_mode="human")
env = safety_gymnasium.make("SafetyCarGoal2-v0", render_mode="human")
# Reset the environment
obs, info = env.reset(seed=1)

s = [f"state {i}" for i in range(env.observation_space.shape[0])]
a = [f"action {i}" for i in range(env.action_space.shape[0])]
l = [f"label"]

col = s + a + l

df = pd.DataFrame(columns=col)
df.to_csv('env.csv', index=False)

while True:
    # Sample a random action
    act = env.action_space.sample()
    # Step the environment: costs are returned
    nxt_obs, reward, cost, terminated, truncated, info = env.step(act)
    sa = np.concatenate((obs, act))
    if cost != 0:
        row = np.append(sa, [1])
        pd_row = pd.DataFrame([row], columns=col)
        print(f"Cost: {cost}")
    else:
        row = np.append(sa, [0])
        pd_row = pd.DataFrame([row], columns=col)
    # if cost != 0.0:
    #     print(f"info: {info}\ncost: {cost}\nreward: {reward}\nAction: {act}")
    # Store the data in the CSV file
    # print(obs)
    # df=df._append(pd_row)
    pd_row.to_csv('env.csv', mode='a', index=False, header=False) 
    obs = nxt_obs
    if terminated or truncated:
        obs, info = env.reset()