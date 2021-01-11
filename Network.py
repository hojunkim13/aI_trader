import torch
import torch.nn as nn
from torch.optim import Adam


class Network(nn.Module):
    def __init__(self, state_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(state_dim, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 )
        
        self.mu = nn.Sequential(nn.Linear(256, 1),
                                nn.Tanh()
                                )
        self.std = nn.Sequential(nn.Linear(256, 1),
                                nn.Softplus()
                                )
        self.value = nn.Linear(256,1)
        self.apply(self._weights_init)
        self.cuda()

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, state):
        x = self.fc1(state)
        mu = (self.mu(x)+1.)/2.
        std = self.std(x)
        value = self.value(x)
        return (mu, std), value