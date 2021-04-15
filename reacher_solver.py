import sys
import os
import torch
import numpy as np
from collections import deque
from agent import ReacherAgent
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# CUR_DIR represents the current directory. Useful for accessing log and checkpoints folder
CUR_DIR = os.path.dirname(os.path.realpath(__file__))

# RUN_NAME represents a specific run. Checkpoint files and Tensorboard logs are saved using the RUN_NAME.
# Helpful for comparing different runs.
RUN_NAME = "Test"

ENV_PATH = "Reacher20_Linux/Reacher.x86_64"   # path to the reacher20 env
SCORE_LIMIT = 32                              # mean score of 100 episodes to reach

# initialize the environment
env = UnityEnvironment(file_name=ENV_PATH)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
num_outputs = brain.vector_action_space_size
print('Size of each action:', num_outputs)

# examine the state space
states = env_info.vector_observations
num_inputs = states.shape[1]

scores = []  # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores

writer = SummaryWriter(log_dir=CUR_DIR + "/log/tensorboard/" + RUN_NAME)  # initialize writer object for tensorboard

# create the ReacherAgent object
agent = ReacherAgent(num_inputs, num_outputs, env, brain_name, num_agents)
i_episode = 0

while True:  # train until the SCORE_LIMIT is not reached
    reward = agent.step()
    if reward is not None:
        scores.append(reward)
        scores_window.append(reward)
        score_mean_100 = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.10f}'.format(i_episode, score_mean_100), end="")

        # write to tensorboard
        writer.add_scalar("score_mean_100", score_mean_100, i_episode)
        writer.add_scalar("score", reward, i_episode)
        writer.flush()

        if i_episode > 0 and i_episode % 100 == 0:
            print()

        # finish training once score limit is reached
        if score_mean_100 > SCORE_LIMIT:
            torch.save(agent.model.state_dict(), 'checkpoints/' + RUN_NAME + '.pth')
            break

        i_episode += 1

