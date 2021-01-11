from Agent import Agent
from Environment import Environment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
matplotlib.rcParams['axes.unicode_minus'] = False


##### params for Environment ###
end = datetime.today()
start = end + timedelta(days = -30)

n_episode = 200
seed = 10
sequence_length = 7
fee = 0.0015 
amp = 10
maginot_line = -100
### params for Agent
path = './model/KOSPI_'
render = False
state_dim = sequence_length * 5
### parms for Network
lr = 1e-4
epsilon = 0.2
gamma = 0.99
lmbda = 0.95
buffer_size = 1000
batch_size = 512
k_epochs = 10



env = Environment(n_episode,seed = seed, sequence_length = sequence_length, fee= fee,
                        amp = amp, maginot_line = maginot_line,
                        start = start,
                        end = end)

agent = Agent(state_dim, lr = lr, epsilon = epsilon, gamma = gamma,
              lmbda = lmbda, buffer_size = buffer_size,
              batch_size= batch_size, k_epochs= k_epochs)
agent.load(path)

def main_test():
    origin_ratio_list = []
    stock_name_list = []
    
    for e in range(n_episode):
        state = env.reset(e)
        done = False
        while not done:
            order, log_prob = agent.get_action(state)
            order = np.round(order, 2)
            state_, reward, done, origin_ratio, pr = env.step(order, render)
            
            state = state_
        #Epi done
        origin_ratio_list.append(profit_ratio)
        stock_name_list.append(env.name)
    #all epi done
    Average_profit = np.mean(origin_ratio_list[-100:])
    count = list(filter(lambda x: x > 0, profit_ratio_list))
    ratio = len(count) / len(origin_ratio_list)
    print('[평균 일일 수익률: {:.1f}, 이득 일수 비율: {:.1f}%]'.format(Average_profit, ratio * 100))
    plt.scatter(stock_name_list, origin_ratio_list)
    #plt.scatter(range(n_episode), profit_ratio_list)
    plt.xticks(rotation = 90)
    plt.ylabel('Profit Ratio [%]')
    plt.show()

def stock_item_test():
    pr_list = []
    time_list = []
    state = env.reset(0)
    done = False
    while not done:
        order, log_prob = agent.get_action(state)
        order = np.round(order, 1)
        state_, reward, done, _,pr = env.step(order, True)
        
        time_list.append(str(env.time)[2:10])
        state = state_
        pr_list.append(pr)
    #all epi done
    Average_profit = np.mean(pr_list)
    count = list(filter(lambda x: x > 0, pr_list))
    ratio = len(count) / len(pr_list)
    print('[평균 일일 수익률: {:.1f}, 이득 일수 비율: {:.1f}%]'.format(Average_profit, ratio * 100))
    order, _ = agent.get_action(state)
    print('[예측 : {:.2f}]'.format(order))
    plt.scatter(time_list, pr_list)
    #plt.scatter(range(n_episode), profit_ratio_list)
    plt.xticks(rotation = 90)
    plt.ylabel('Profit Ratio [%]')
    plt.title(env.name)
    #plt.show()
    

if __name__ == "__main__":
    stock_item_test()