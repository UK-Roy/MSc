import safety_gymnasium
from stable_baselines3 import A2C

env = safety_gymnasium.vector.make("SafetyCarGoal1-v0", render_mode="human")
observation, info = env.reset(seed=0)

# Added
# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)

# vec_env = model.get_env()
# observation, info = vec_env.reset(seed=0)
# End added

ep_ret, ep_cost = 0, 0
for _ in range(1000):
   # action, _ = model.predict(observation, deterministic=True)
   # observation, reward, cost, terminated, truncated, info = env.step(action)
   
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, cost, terminated, truncated, info = env.step(action)

   ep_ret += reward
   ep_cost += cost

   if terminated or truncated:
      observation, info = env.reset()

print(ep_ret)
print(ep_cost)
env.close()