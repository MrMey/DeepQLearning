import numpy as np
import random


class SQL:
    def __init__(self,
                 state_size,
                 action_size,
                 lr=0.15,
                 discount_rate=0.99,
                 max_exploration_rate=1,
                 min_exploration_rate=0.0001,
                 exploration_discount=0.999
                 ):
        self.action_size = action_size
        self.state_size = state_size

        self.lr = lr
        self.discount_rate = discount_rate

        self.exploration_rate = max_exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_discount = exploration_discount
        self._build_model()

    def _build_model(self):
        self.q_table = np.zeros((self.state_size, self.action_size))

    def act(self, state):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > self.exploration_rate:
            action = np.argmax(self.q_table[state, :])
        else:
            action = random.randrange(self.action_size)
        return action

    def remember(self, state, action, reward, next_state, done):
        # discount previous experiments
        self.q_table[state, action] *=  (1 - self.lr)
        # add futur gains to the current reward
        reward += self.discount_rate * np.max(self.q_table[next_state, :])
        # update the table
        self.q_table[state, action] += self.lr * reward

        self.exploration_rate *= self.exploration_discount
