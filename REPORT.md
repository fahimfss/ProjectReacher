# Report on Project Reacher

## Overview of the Algorithm

In this project, I used the Actor Critic Algorithm with GAE, for solving the Unity Reacher environment.  

The training of the agent takes place in the [reacher_solver.py](https://github.com/fahimfss/ProjectReacher/blob/master/reacher_solver.py) file. Here's a very basic overview of the Algorithm for training the agent:
- Initially, the Unity Reacher environment is initialized. Reacher20 was used in this project, which is a collection of 20 Reacher environments. This environment is responsible for providing the state, reward, next-state and done (if an episode is completed) values.
- Then, an agent object is created which is responsible for selecting an action based on the current state. In Actor Critic, the agent uses a Deep Neural Network for action selection. In this project, the agent codes are written in the Agent class [(agent.py)](https://github.com/fahimfss/ProjectReacher/blob/master/agent.py) and the DNN codes are written in the A2CNetwork class [(model.py)](https://github.com/fahimfss/ProjectReacher/blob/master/model.py)
- The agent picks an action based on the current state provided by the environment. Based on the action, the environment provides next-state, reward, and done values. This process is repeated for a very long time. 
- To choose better actions, the agent needs to learn by using the values provided by the environment. The Actor Critic algorithm used in this project is a policy gradient algorithm. That means the agent learns the state to action maping directly instead of learning state values or state action values. Actor Critic is also an on-policy algorithm, so we can not use replay memory here. To learn, the agent performs a 10 step rollout on the environment, calculates the return values of each step using GAE and performs a gradient update to increase the probabilities of good actions.
- The Actor Critic algorithm also uses a concept called baseline. Baseline is removed from the return values to decrease variance. In this project, the state value, V(S), is used as baseline. During training the agent also learns to predict the baseline value, V(S), correctly.   
- After the training reaches a certain level (in this environment, when the mean reward reaches the value 32 for the last 100 episodes), the training is finished.

#### Hyperparameters
**Solver ([reacher_solver.py](https://github.com/fahimfss/ProjectReacher/blob/master/reacher_solver.py)):** state_size=33, action_size=4, SCORE_LIMIT=32   
**Agent ([agent.py](https://github.com/fahimfss/ProjectReacher/blob/master/agent.py)):** NUM_STEPS=10, LR=1e-4

## Implementation details and contribution
The Actor Critic GAE code in the project is based on the codes in [higgsfield/RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2/blob/master/2.gae.ipynb). I made several changes to that code in order to solve the Reacher environment. Those are stated below:
- I changed the deep neural network architecture for predicting actor and critic values. I used a bigger network with more hidden layers. Here's a summary of the deep neural network that was used: 
![A2CNetwork](https://user-images.githubusercontent.com/8725869/114942043-c3469380-9e65-11eb-839a-8192c87cb62b.png)
- I used Adam optimizer with learning rate 1e-4 and eps 1e-3
- I normalized the state space by dividing it with 10.0 (line 93 of agent.py file)
- I increased the reward values given to the agents by 20 (line 94 of agent.py file). The provided reward values were too small, especially during the early episodes. But for score calculation and log, I used the actual reward values. 
  
After applying these optimizations, I was able to solve the Reacher environment.  
  
I would like to add that, the compute_gae method (line 44 of agent.py file) is crucial for solving the environment. My notes on the compute_gae method can be viewed below:  

<details><summary><b>Click to view my notes on compute_gae method</b></summary>
  
![compute_gae](https://user-images.githubusercontent.com/8725869/114943264-8aa7b980-9e67-11eb-8e8d-76153eeb466c.png)
</details>

## Results
The code in its current state was able to achieve a mean score of 32 over 100 episodes in three different random runs. Here's a plot of the mean reward over 100 episodes vs episode number for the three runs:  
![sc2](https://user-images.githubusercontent.com/8725869/114943702-38b36380-9e68-11eb-8d32-7048f1c9e258.png)
  
The following table contains the summary of the three runs:  
|Run Name (Color)|Episodes to reach mean rewards of 32|Time taken to reach mean rewards of 32|
|:-------|:----------------------------------:|:----------------------------------:|
|Test1 (Orange)|286|24m 06s|
|Test2 (Blue)|341|28m 34s|
|Test3 (Red)|209|17m 27s|

This plot is created using tensorboard, with log files located at "[/log/tensorboard](https://github.com/fahimfss/ProjectReacher/tree/master/log/tensorboard)". All the runs successfully converged, but there is a significant performance difference between Test1 run and Test3 run. The performance difference between different runs is normal for policy gradient algos.   

Here's a video of a trained agent collecting bananas in the environment:  

[Test3](https://user-images.githubusercontent.com/8725869/114944394-53d2a300-9e69-11eb-82f3-b9080e51c8d7.mp4)


This video was created by running the [Test3](https://github.com/fahimfss/ProjectReacher/tree/master/checkpoints) agent, using the [run.py](https://github.com/fahimfss/ProjectReacher/blob/master/run.py) file.  

## Future Works
- To solve the environment by implementing the PPO and DDPG algorithm.  
- To solve the Crawler environment
- I implemented an RL agent to solve my own game before ([Meteors](https://github.com/fahimfss/RL/tree/master/DQN)). I will improve on that project by applying the knowledge learned from this project. 
