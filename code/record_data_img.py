import os
import numpy as np
import json
from PIL import Image

import safety_gymnasium
from gymnasium.utils.save_video import save_video

directory_name = "SafetyCarGoalDataset"
num_episode = 5

# Create the safety-task environment
# env = safety_gymnasium.make("SafetyCarGoal1-v0", render_mode="human")
env = safety_gymnasium.make("SafetyCarGoal2VisionDebug-v0")

os.makedirs(directory_name, exist_ok=True)
print(f"Directory '{directory_name}' created successfully.")
os.chdir(directory_name)

episode = 1

while episode <= num_episode:
    
    # Reset the environment
    obs, info = env.reset(seed=1)
    terminated, truncated = False, False
    
    sub_directory = f"{episode}"
    os.makedirs(sub_directory, exist_ok=True)
    os.chdir(sub_directory)
    rollout = 1
    render_list = []

    while not terminated and not truncated:
        
        rollout_dir = f"{rollout}"
        os.makedirs(rollout_dir, exist_ok=True)
        os.chdir(rollout_dir)

        # Sample a random action
        act = env.action_space.sample()
        data = {}        
        render_list.append(obs['vision'])
         
        # Step the environment: costs are returned
        nxt_obs, reward, cost, terminated, truncated, info = env.step(act)
        # sa = np.concatenate((obs, act))
        for key, value in obs.items():
            if isinstance(obs[key], np.ndarray):
                data[key] = obs[key].tolist()
            else:
                data[key] = obs[key]
            # print(f"Key: {key}, Value: {value}")
        
        data['action'] = act.tolist()
        if cost != 0:
            data['label'] = f"unsafe"
        else:
            data['label'] = f"safe"
        
        with open('state.json', 'w') as file:
            json.dump(data, file, indent=4)
        
        image = Image.fromarray(obs['vision'])
        image.save(f'{rollout}.png')

        obs = nxt_obs
        
        rollout += 1
        os.chdir('..')
    
    os.chdir('..')
    episode += 1
    save_video(
        frames=render_list,
        video_folder=sub_directory,
        name_prefix='render_vision_output',
        fps=30,
    )

print("Dataset Completed")