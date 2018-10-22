from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

# Deep Q-learning Agent
class DQNAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 batch_size = 32):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1,self.state_size])
        next_state = np.reshape(next_state, [1,self.state_size])
        
        self.memory.append((state, action, reward, next_state, done))
        self._replay()

    def act(self, state):
        state = np.reshape(state, [1,self.state_size])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def _replay(self):
        if len(self.memory) < self.batch_size:
            # not enough experiments in memory to replay
            return
        
        # sample batch_size experiments from memory
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            # if terminal action then assign reward
            target = reward
            # if not terminal then add the discounted future reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # predict the policies reward
            target_f = self.model.predict(state)

            # the reward for the chosen policy is the one calculated
            # the others rewards are left unchanged
            target_f[0][action] = target

            # train the model with the rewards
            self.model.fit(state, target_f, epochs=1, verbose=0)

        # decay the exploration rate        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
