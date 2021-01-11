from Agent import Agent
from Environment import Environment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['axes.unicode_minus'] = False


##### params for Environment ###
n_episode = 1000
seed = 10
sequence_length = 7
fee = 0.0015
amp = 10.
maginot_line = -30
### params for Agent
path = './model/KOSPI_'
load = False
render = False
save_cycle = 10
state_dim = sequence_length * 5
### parms for Networkd
lr = 1e-4
epsilon = 0.2
gamma = 0.99
lmbda = 0.95
buffer_size = 1000
batch_size = 256
k_epochs = 10



env = Environment(n_episode,seed = seed, sequence_length = sequence_length, fee= fee,
                        amp = amp, maginot_line = maginot_line)

agent = Agent(state_dim, lr = lr, epsilon = epsilon, gamma = gamma,
              lmbda = lmbda, buffer_size = buffer_size,
              batch_size= batch_size, k_epochs= k_epochs)

if __name__ == "__main__":
    if load:
        agent.load(path)

    profit_ratio_list = []
    stock_name_list = []
    
    for e in range(n_episode):
        state = env.reset(e)
        done = False
        while not done:
            order, log_prob = agent.get_action(state)
            order = np.round(order, 2)
            
            state_, reward, done, profit_ratio,pr = env.step(order, render)
            agent.store((state, order, log_prob, reward, state_, done))
            agent.learn()
            state = state_
        #Epi done
        profit_ratio_list.append(profit_ratio)
        stock_name_list.append(env.name)
        if (e+1) % save_cycle ==0:
            agent.save(path)
    #all epi done
    Average_profit = np.mean(profit_ratio_list[-100:])
    count = len(list(filter(lambda x: x > 0, profit_ratio_list)))
    print('[평균 수익률: {:.1f}%, 손익률: {:.0f}%]'.format(Average_profit, count/n_episode * 100))
    plt.scatter(stock_name_list, profit_ratio_list)
    #plt.scatter(range(n_episode), profit_ratio_list)
    plt.xticks(rotation = 90)
    plt.ylabel('Profit Ratio')
    plt.show()