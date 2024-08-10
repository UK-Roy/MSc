import numpy as np
import random

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # self.Q_values = np.zeros((state_size, action_size))
        BOARD_ROWS = state_size[0]
        BOARD_COLS = state_size[1]
        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in range(self.action_size):
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.Q_values[state[:2]])
    
    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q_values[next_state[:2]])
        td_target = reward + self.discount_factor * self.Q_values[next_state[:2]][best_next_action]
        td_error = td_target - self.Q_values[state[:2]][action]
        self.Q_values[state[:2]][action] += self.learning_rate * td_error