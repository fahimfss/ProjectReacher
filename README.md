# Udacity Deep Reinforcement Learning Project: Reacher
## Project Overview 
 In this project, I experimented with the Unity Reacher environment. I used Actor Critic algorithm in my project, along with Generalized Advantage Estimation (GAE) [(link)](https://arxiv.org/abs/1506.02438).

#### Project Files
- **reacher_sovler.py:**  This file contains the the code which is used to train the RL agent  
- **agent.py:**  This file contains the Agent class, which is responsible for interacting with the environment and train the neural network 
responsible for selecting actions based on different states.
- **model.py:** This file contains the Neural Network architecture used by the agent.
- **run.py:** This file can run a trained Agent on the Reacher environment 
- **log/tensorboard:** This folder contains the tensorboard graphs of different training runs
- **checkpoints:** This folder contains saved models of different runs
<br/>

Every RL project should have well-defined state, action and reward spaces. For this project the state, action and reward spaces are described below:  
- **State-space:** State-space is an array representation of the Reacher environment consisting of 33 floating-point values.  
- **Action-space:** Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector is a floating-point number in the range [-1, 1].    
- **Agent's goal:** The agent's goal is to maximize the reward by staying in the target position as long as possible. The target position might move or stay still. In this project, the environment is considered solved, when the agent is capable of collecting 32 rewards on average for the last 100 episodes.
<br/>

## Getting Started
- The following python libraries are required to run the project: pytorch, numpy, tensorboardx and unityagents
- The Reacher environment folder is not included in this github project, but can be found [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip).
<br/>

## Instructions
#### To Train the Agent
To train the agent, all the files and folders mentioned in the **Project Files**, should be saved in a directory. Then the **reacher_sovler.py** file should 
be run using a python 3 interpreter. Two things to note while running the project for training:
- The **reacher_sovler.py** assumes that the Reacher environment is in the same directory as itself. The location of the 
Reacher environment directory can be updated in line no 19 of the **reacher_sovler.py** file. 
- The RUN_NAME (line 17 of **reacher_sovler.py**) corresponds to a specific run, and creates a tensordboard graph and checkpoint file with the given value.
Different runs should have different RUN_NAME values.
  
#### To Run a Trained Agent
Trained agents (network state dictionaries) are stored in the checkpoints folder, containing the name ***RUN_NAME***.pth. Trained means the agent achieved 
average reward of 32 for 100 episodes in the Reacher environment. The checkpoints folder contains three trained agents: Test1.pth, Test2.pth, Test3.pth.
To run a trained agent, update the RUN_NAME in the **run.py** file (line 10) and run the **run.py** file using a python 3 interpreter. The path to the Reacher environment might need to be updated also (line 11). 
<br/>

## Results
Please check the [report](https://github.com/fahimfss/ProjectReacher/blob/master/REPORT.md) file for the implementation and result details.
