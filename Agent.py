from Network import Network
import numpy as np
import torch
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



class Agent:
    def __init__(self, state_dim, lr, epsilon, gamma, lmbda,  buffer_size, batch_size, k_epochs):
        self.net = Network(state_dim)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.k_epochs = k_epochs
        self.state_dim = state_dim
        self.action_dim = 1
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.S = np.zeros((buffer_size, state_dim), dtype = 'float')
        self.A = np.zeros((buffer_size, 1), dtype = 'float')
        self.P = np.zeros((buffer_size, 1), dtype = 'float')
        self.R = np.zeros((buffer_size, 1), dtype = 'float')
        self.S_= np.zeros((buffer_size, state_dim), dtype = 'float')
        self.D = np.zeros((buffer_size, 1), dtype = 'bool')
        self.mntr = 0                                 
        
    def get_action(self, state):
        state = torch.Tensor(state).cuda()
        with torch.no_grad():
            mu, std = self.net(state)[0]
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()[0]
        log_prob = dist.log_prob(action)[0]
        return action.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def store(self, transition):
        index = self.mntr % self.buffer_size
        self.S[index] = transition[0]
        self.A[index] = transition[1]
        self.P[index] = transition[2]
        self.R[index] = transition[3]
        self.S_[index] = transition[4]
        self.D[index] = transition[5]
        self.mntr += 1

    def learn(self):
        if self.mntr != self.buffer_size:
            return          
        S = torch.Tensor(self.S).float().cuda()
        A = torch.Tensor(self.A).float().cuda()
        log_prob_old = torch.Tensor(self.P).float().cuda()
        R = torch.Tensor(self.R).float().cuda()
        S_ = torch.Tensor(self.S_).float().cuda()
        D = torch.BoolTensor(self.D).cuda()
        
        td_target, advantage = self.get_advantage(S,R,S_,D)

        for i in range(self.k_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_size)), self.batch_size, False):
                (mu, std), value = self.net(S[index])                
                dist = torch.distributions.Normal(mu, std)
                log_prob_new = dist.log_prob(A[index])
                ratio = torch.exp(log_prob_new - log_prob_old[index])
                surrogate1 = ratio * advantage[index]
                surrogate2 = torch.clip(ratio, 1-self.epsilon, 1+self.epsilon) * advantage[index]
                a_loss = -torch.min(surrogate1, surrogate2).mean()
                v_loss = F.smooth_l1_loss(value, td_target[index])
                
                self.optimizer.zero_grad()
                (a_loss + v_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
                self.mntr = 0

    def get_advantage(self, S, R, S_, D):
        with torch.no_grad():
            td_target = R + self.gamma * self.net(S_)[1] * ~D
            delta = td_target - self.net(S)[1]
        advantage = torch.zeros_like(delta)
        running_add = 0
        for i in reversed(range(len(delta))):
            advantage[i] = delta[i] + running_add * self.gamma * self.lmbda
            running_add = advantage[i]
        return td_target, advantage

    def save(self, path):
        torch.save(self.net.state_dict(), path + '.pt')

    def load(self, path):
        self.net.load_state_dict(torch.load(path + '.pt'))
