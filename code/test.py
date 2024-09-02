
import argparse
import os
import json
from collections import deque
# from safepo.common.env import make_sa_mujoco_env, make_ma_mujoco_env, make_ma_multi_goal_env, make_sa_isaac_env
from safepo.common.model import ActorVCritic
# from safepo.utils.config import multi_agent_velocity_map, multi_agent_goal_tasks
import numpy as np
import joblib
import torch
import safety_gymnasium

from safety_gymnasium.wrappers import SafeAutoResetWrapper, SafeRescaleAction, SafeUnsqueeze
from safepo.common.wrappers import SafeNormalizeObservation


def eval_single_agent(eval_dir, eval_episodes):

    torch.set_num_threads(4)
    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))

    env_id = config['task'] if 'task' in config.keys() else config['env_name']
    env_norms = os.listdir(eval_dir)
    env_norms = [env_norm for env_norm in env_norms if env_norm.endswith('.pkl')]
    final_norm_name = sorted(env_norms)[-2]

    model_dir = eval_dir + '/torch_save'
    models = os.listdir(model_dir)
    models = [model for model in models if model.endswith('.pt')]
    final_model_name = sorted(models)[-2]

    model_path = model_dir + '/' + final_model_name
    norm_path = eval_dir + '/' + final_norm_name

    # eval_env, obs_space, act_space = make_sa_mujoco_env(num_envs=config['num_envs'], env_id=env_id, seed=None)

    env = safety_gymnasium.make(env_id, render_mode='human')
    env.reset()
    obs_space = env.observation_space
    act_space = env.action_space
    env = SafeAutoResetWrapper(env)
    env = SafeRescaleAction(env, -1.0, 1.0)
    env = SafeNormalizeObservation(env)
    env = SafeUnsqueeze(env)
    eval_env = env

    model = ActorVCritic(
            obs_dim=obs_space.shape[0],
            act_dim=act_space.shape[0],
            hidden_sizes=config['hidden_sizes'],
        )
    model.actor.load_state_dict(torch.load(model_path))

    if os.path.exists(norm_path):
        norm = joblib.load(open(norm_path, 'rb'))['Normalizer']
        eval_env.obs_rms = norm

    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)

    for _ in range(eval_episodes):
        eval_done = False
        eval_obs, _ = eval_env.reset()
        eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32)
        eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
        while not eval_done:
            with torch.no_grad():
                act, _, _, _ = model.step(
                    eval_obs, deterministic=True
                )
            eval_obs, reward, cost, terminated, truncated, info = eval_env.step(
                act.detach().squeeze().cpu().numpy()
            )
            eval_obs = torch.as_tensor(
                eval_obs, dtype=torch.float32
            )
            eval_rew += reward[0]
            eval_cost += cost[0]
            eval_len += 1
            eval_done = terminated[0] or truncated[0]
            eval_env.render()
        eval_rew_deque.append(eval_rew)
        eval_cost_deque.append(eval_cost)
        eval_len_deque.append(eval_len)

    return sum(eval_rew_deque) / len(eval_rew_deque), sum(eval_cost_deque) / len(eval_cost_deque)

def single_runs_eval(eval_dir, eval_episodes):

    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))
    env = config['task'] if 'task' in config.keys() else config['env_name']
    reward, cost = eval_single_agent(eval_dir, eval_episodes)
    
    return reward, cost

def benchmark_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=str, default='safepo\\runs\\single_agent_exp', help="the directory of the evaluation")
    parser.add_argument("--eval-episodes", type=int, default=3, help="the number of episodes to evaluate")
    parser.add_argument("--save-dir", type=str, default='safepo\\results\\ppo_lag_exp', help="the directory to save the evaluation result")

    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir
    eval_episodes = args.eval_episodes
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = benchmark_dir.replace('runs', 'results')
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
    envs = os.listdir(benchmark_dir)
    for env in envs:
        env_path = os.path.join(benchmark_dir, env)
        algos = os.listdir(env_path)
        for algo in algos:
            print(f"Start evaluating {algo} in {env}")
            algo_path = os.path.join(env_path, algo)
            seeds = os.listdir(algo_path)
            rewards, costs = [], []
            for seed in seeds:
                seed_path = os.path.join(algo_path, seed)
                reward, cost = single_runs_eval(seed_path, eval_episodes)
                rewards.append(reward)
                costs.append(cost)
            output_file = open(f"{save_dir}\\eval_result.txt", 'a')
            # two wise after point
            reward_mean = round(np.mean(rewards), 2)
            reward_std = round(np.std(rewards), 2)
            cost_mean = round(np.mean(costs), 2)
            cost_std = round(np.std(costs), 2)
            print(f"After {eval_episodes} episodes evaluation, the {algo} in {env} evaluation reward: {reward_mean}±{reward_std}, cost: {cost_mean}±{cost_std}, the reuslt is saved in {save_dir}/eval_result.txt")
            output_file.write(f"After {eval_episodes} episodes evaluation, the {algo} in {env} evaluation reward: {reward_mean}±{reward_std}, cost: {cost_mean}±{cost_std} \n")

if __name__ == '__main__':
    benchmark_eval()
