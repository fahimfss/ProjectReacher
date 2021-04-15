import torch
import torch.optim as optim
import numpy as np
from model import A2CNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters:
NUM_STEPS = 10  # number of steps in rollout
LEARNING_RATE = 1e-4  # learning rate for the optimizer


class ReacherAgent:
    """ The ReacherAgent class is responsible for interacting with the reacher environment and
        teaching the neural network (model) to take appropriate actions based on states

        This class is based on codes found here: https://github.com/higgsfield/RL-Adventure-2/blob/master/2.gae.ipynb
        """

    def __init__(self, num_inputs, num_outputs, env, brain_name, num_agents):
        """Initialize an ReacherAgent object.

        Params
        ======
            num_inputs (int): dimension of each state
            num_outputs (int): dimension of each action
            env: the reacher environment object
            brain_name: reacher environment brain name
            num_agents: number of agents in reacher environment
        """

        self.model = A2CNetwork(num_inputs, num_outputs).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, eps=1e-3)
        self.env = env
        self.brain = brain_name
        self.num_agents = num_agents

        # Initial state is stored in self.state. The states are normalized by dividing by 10.0
        env_info = self.env.reset(train_mode=True)[self.brain]
        self.state = env_info.vector_observations / 10.0

        self.scores = np.zeros(self.num_agents)  # scores keeps track of the env rewards

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        """ Returns the GAE (Generalized advantage estimation) value for a rollout,
        check "images/compute_gae.png" for understanding how this method works

        Params
        ======
            next_value: state value of the the last state in rollout
            rewards: reward values during rollout
            masks: terminal state masks of each state in rollout (1 means that state is terminal)
            values: state values of each state in rollout
            gamma: discount rate
            tau: exponentially decaying factor for calculating GAE
        """

        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def step(self):
        """ step method does a NUM_STEPS rollout in the environment and updates the neural network (model)

            :returns
            The episode number and the mean score of agents in that episode if an episode ends in rollout
        """

        log_probs = []                      # array to store log probabilities of each action in rollout
        values = []                         # array to store state values (from model) of each state in rollout
        rewards = []                        # array to store reward values for each step in rollout (provided by env)
        masks = []                          # array to store terminal state flags for each state in rollout
        # Size of all four arrays after rollout: [NUM_STEPS, NUM_AGENTS]

        entropy = 0                         # entropy term to encourage exploration

        ret = None

        for _ in range(NUM_STEPS):          # NUM_STEPS rollout

            # Select action using the current state
            state = torch.FloatTensor(self.state).to(device)
            dist, value = self.model(state)       # the model returns a normal distribution for action value, and V(s)
            action = dist.sample().clamp(-1, 1)   # action is sampled from the distribution and clipped in range [-1, 1]
            action_np = action.cpu().numpy()

            env_info = self.env.step(action_np)[self.brain]      # action is applied in the env
            next_state = env_info.vector_observations / 10.0     # next_state is normalized by dividing by 10.0
            reward = np.asarray(env_info.rewards) * 20.0         # rewards are amplified by multiplying by 20
            done = np.array(env_info.local_done, dtype=int)      # done (terminal) flags are created

            self.scores += env_info.rewards      # actual reward values are added to score

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            # related values are added to the arrays
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            self.state = next_state

            if np.any(done):    # for the given env, all the 20 environments terminate on the same frame
                env_info = self.env.reset(train_mode=True)[self.brain]    # reset the env
                self.state = env_info.vector_observations
                ret = self.scores.mean()                      # prepare return value, which is the mean of actual scores
                self.scores = np.zeros(self.num_agents)       # reset scores

        # state value of the last state is calculated using the neural network (model)
        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = self.model(next_state)

        # using the prepared arrays, the return values are calculated
        returns = self.compute_gae(next_value, rewards, masks, values)

        # convert arrays to torch tensors
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        # calculate the advantage value
        advantage = returns - values

        # calculate actor loss (negative because of gradient ascend)
        actor_loss = -(log_probs * advantage.detach()).mean()

        # calculate critic loss
        critic_loss = advantage.pow(2).mean()

        # calculate total loss
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        # calculate gradients and update the neural network (model)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ret

    def act(self, state):
        # given a state, returns the corresponding action

        state = state / 10.0
        state = torch.FloatTensor(state).to(device)
        dist, value = self.model(state)  # the model returns a normal distribution for action value, and V(s)
        action = dist.sample().clamp(-1, 1)  # action is sampled from the distribution and clipped in range [-1, 1]
        action_np = action.cpu().numpy()
        return action_np

