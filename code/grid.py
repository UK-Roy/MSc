import numpy as np

class GridWorldEnv:
    def __init__(self, grid_size=5, human_position=(2, 2), goal_position=(4, 4)):
        self.grid_size = grid_size
        self.human_position = np.array(human_position)
        self.goal_position = np.array(goal_position)
        self.reset()
    
    def reset(self):
        self.robot_position = np.array([0, 0])
        self.state = np.concatenate([self.robot_position, self.human_position])
        return self.state
    
    def step(self, action):
        if action == 0:  # left
            self.robot_position[0] -= 1
        elif action == 1:  # right
            self.robot_position[0] += 1
        elif action == 2:  # up
            self.robot_position[1] += 1
        elif action == 3:  # down
            self.robot_position[1] -= 1
        
        self.robot_position = np.clip(self.robot_position, 0, self.grid_size-1)
        self.state = np.concatenate([self.robot_position, self.human_position])
        
        reward = self.task_reward()
        done = np.array_equal(self.robot_position, self.goal_position)
        
        energy_cost = self.energy_function()
        reward -= energy_cost
        
        return self.state, reward, done
    
    def task_reward(self):
        if np.array_equal(self.robot_position, self.goal_position):
            return 10
        else:
            return 0
    
    def energy_function(self):
        robot_position = self.state[:2]
        human_position = self.state[2:4]
        distance_to_human = np.linalg.norm(robot_position - human_position)
        E_proximity = 1 / distance_to_human if distance_to_human != 0 else float('inf')
        E_control = np.sum(np.abs(robot_position - self.state[:2]))
        total_energy = E_proximity + E_control
        return total_energy
    
    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.human_position[1], self.human_position[0]] = 1
        grid[self.goal_position[1], self.goal_position[0]] = 0.5
        grid[self.robot_position[1], self.robot_position[0]] = 0.75
        print(grid)

