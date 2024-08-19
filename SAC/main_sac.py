import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import safety_gymnasium

if __name__ == '__main__':
    env = safety_gymnasium.vector.make("SafetyCarGoal1-v0")
    # env = safety_gymnasium.vector.make("SafetyCarGoal1-v0", render_mode="human")
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[1])
    n_games = 250
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'navigation.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation, info = env.reset(seed=0)
        terminated = False
        truncated = False
        score = 0
        step = 0
        while not terminated and not truncated and step<1000:
            action = agent.choose_action(observation)
            observation_, reward, cost, terminated, truncated, info= env.step(action[0])
            score += reward
            agent.remember(observation, action, reward, observation_, terminated)
            if not load_checkpoint:
                agent.learn()
                # print(f"Learning... ):")
            observation = observation_
            step += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

