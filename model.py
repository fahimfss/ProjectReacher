import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def init_weights(m):
    """ Initializes the weights and bias values of the Linear layers in the network """

    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class A2CNetwork(nn.Module):
    def __init__(self, state_size, action_size, std=0.0):
        super(A2CNetwork, self).__init__()

        # the critic network,
        # it's output represents the state value for baseline
        self.critic = nn.Sequential(
            nn.Linear(state_size, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

        # the actor network
        self.actor = nn.Sequential(
            nn.Linear(state_size, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_size),
        )

        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)

        self.apply(init_weights)

    def forward(self, x):
        """ Returns distributions of actions and state value for input state """

        # get the output of the critic network (state value)
        value = self.critic(x)

        # get the output of the actor network, in the range -1 to 1
        mu = torch.tanh(self.actor(x))

        # calculating the standard deviation for the normal distribution
        # if self.log_std is zeros tensor, then std is ones tensor
        std = self.log_std.exp().expand_as(mu)

        # creating a normal distribution using mu and std
        # using this normal distribution (dist), the action value will be sampled
        dist = Normal(mu, std)

        return dist, value

