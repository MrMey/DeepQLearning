import gym
import time
import datetime
# project modules
import dqn
import sql

"""
Given a cartpole environment : try different Q-learning techniques and measure performance
"""
# Experiment parameters:
EPISODES = 10000
STEPS = 500
RENDER = False
ENV_NAME = "CartPole-v0"
AGENT_NAME = "dqn"

# create path for the experiment
exp_path = "results/" + ENV_NAME + "_" + AGENT_NAME 
exp_path += "_" + datetime.datetime.now().strftime(format="%m_%d_%H_%M")

def run_experiment():
    env = gym.make(ENV_NAME)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = dqn.DQNAgent(state_size,action_size)

    for e in range(EPISODES):
        score = 0
        state = env.reset()
        start = time.time()
        for s in range(STEPS):
            if RENDER:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                elapse = time.time()-start
                print("episode: {}/{}, score: {}, elapsed : {}"
                      .format(e, EPISODES, score, elapse))
                break
            score += reward
        
        with open(exp_path + ".txt", "a") as f:
            f.write(str(e) + ',' + str(score) + "\n")

if __name__ == "__main__":
    run_experiment()