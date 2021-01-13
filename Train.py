from Agent import Agent
from Environment import Environment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
matplotlib.rcParams['axes.unicode_minus'] = False


##### params for Environment ###
n_episode = 200
sequence_length = 7
amp = 10
clip = int(np.log10(amp))
filter_item = True
### params for Agent
path = './model/KOSPI_'
load = False
render = False
save_cycle = 10
state_dim = sequence_length * 5
### parms for Network
lr = 1e-5
epsilon = 0.2
gamma = 0.5
lmbda = 0.5
buffer_size = 1000
batch_size = 512
k_epochs = 10



env = Environment(n_episode = n_episode,
                  sequence_length = sequence_length,
                  amp = amp)

agent = Agent(state_dim, lr = lr, epsilon = epsilon, gamma = gamma,
              lmbda = lmbda, buffer_size = buffer_size,
              batch_size= batch_size, k_epochs= k_epochs)

if __name__ == "__main__":
    if load:
        agent.load(path)

    stock_name_list = []
    score_list = []
    for e in range(n_episode):
        state = env.reset(e)
        done = False
        loss_count = 0
        step_count = 0
        while not done:
            order, log_prob = agent.get_action(state)
            order = np.round(order, clip)
            state_, reward, done, profit_list = env.step(order, render)
            agent.store((state, order, log_prob, reward, state_, done))
            agent.learn()
            state = state_
        #Epi done
        score_list.append(np.mean(profit_list))
        stock_name_list.append(env.name)
        if (e+1) % save_cycle ==0:
            agent.save(path)
    #all epi done
    ''' n = pd.DataFrame(name_list, columns = ['Name'])
    c = pd.DataFrame(code_list, columns = ['Code'])
    value_items = pd.concat([n, c], axis = 1)
    if filter_item:
        value_items.to_csv('./data/train_item.csv', sep=',', encoding = 'utf-8-sig', index = False)'''
    count = len(list(filter(lambda x: x > 0, score_list)))
    total_average = np.mean(score_list)
    print('[총 평균 수익률: {:.1f}%, 이익 종목 비율: {:.0f}%]'.format(total_average, count/n_episode * 100))
    plt.scatter(stock_name_list, score_list)
    plt.xticks(rotation = 90)
    plt.ylabel('Profit Ratio')
    plt.show()
    