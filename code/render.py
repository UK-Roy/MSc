import safety_gymnasium

# Create the safety-task environment
# env = safety_gymnasium.make("SafetyCarGoal1-v0", render_mode="human")
env = safety_gymnasium.make("SafetyCarGoal1-v0")
# Reset the environment
obs, info = env.reset(seed=1)

while True:
    # Sample a random action
    act = env.action_space.sample()
    # Step the environment: costs are returned
    nxt_obs, reward, cost, terminated, truncated, info = env.step(act)
    obs = nxt_obs
    if terminated or truncated:
        obs, info = env.reset()