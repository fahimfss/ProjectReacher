import torch
from agent import ReacherAgent
from unityagents import UnityEnvironment
import numpy as np
import time


# RUN_NAME represents a specific run. Checkpoint files and Tensorboard logs are saved using the RUN_NAME.
# Helpful for comparing different runs.
RUN_NAME = "Test3"
ENV_PATH = "Reacher20_Linux/Reacher.x86_64"   # path to the reacher20 env

# initialize the environment
env = UnityEnvironment(file_name=ENV_PATH)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
num_outputs = brain.vector_action_space_size
print('Size of each action:', num_outputs)

# examine the state space
state = env_info.vector_observations
num_inputs = state.shape[1]

agent = ReacherAgent(num_inputs, num_outputs, env, brain_name, num_agents)
agent.model.load_state_dict(torch.load('checkpoints/' + RUN_NAME + '.pth'))

score = 0  # initialize the score

sleep = 5  # Used for slowing down the environment for recording

while True:
    action = agent.act(state)  # select an action using the trained agent
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations  # get the next state
    reward = env_info.rewards  # get the reward
    done = env_info.local_done  # see if episode has finished
    score += np.mean(reward)  # update the score

    if sleep == 5:
        time.sleep(sleep)
        sleep = 0.0015
    else:
        time.sleep(sleep)

    state = next_state  # roll over the state to next time step
    if np.any(done):  # exit loop if episode finished
        break

