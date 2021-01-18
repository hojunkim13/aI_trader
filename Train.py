from Agent import Agent
from Environment import Environment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime
matplotlib.rcParams['axes.unicode_minus'] = False


##### params for Environment ###
n_episode = 1000
sequence_length = 7
amp = 10
### params for Agent
path = './model/total'
load = True
render = False
save_cycle = 5
state_dim = sequence_length * 10
### parms for Network
lr = 1e-5
epsilon = 0.2
gamma = 0.5
lmbda = 0.5
buffer_size = 5000
batch_size = 512
k_epochs = 30



env = Environment(n_episode = n_episode,
                  sequence_length = sequence_length,
                  amp = amp,
                  )
#env.choice_stock = 124
agent = Agent(state_dim, lr = lr, epsilon = epsilon, gamma = gamma,
              lmbda = lmbda, buffer_size = buffer_size,
              batch_size= batch_size, k_epochs= k_epochs)

if __name__ == "__main__":
    if load:
        agent.load(path)

    stock_name_list = []
    score_list = []
    for e in range(n_episode):
        #state = env.reset(e)
        state = env.reset(e,)
        done = False
        loss_count = 0
        step_count = 0
        score = 0
        orders = []
        while not done:
            order, log_prob = agent.get_action(state)
            state_, reward, done, profit_list = env.step(order, render)
            score += reward
            orders.append(order)
            agent.store((state, order, log_prob, reward, state_, done))
            agent.learn()
            state = state_
        #Epi done
        print(score,np.mean(orders))
        score_list.append(np.mean(profit_list))
        stock_name_list.append(env.name)
        if (e+1) % save_cycle ==0:
            agent.save(path)
    #all epi done

    count = len(list(filter(lambda x: x > 0, score_list)))
    total_average = np.mean(score_list)
    print('[총 평균 수익률: {:.2f}%, 이익 종목 비율: {:.0f}%]'.format(total_average, count/n_episode * 100))
    plt.scatter(stock_name_list, score_list)
    plt.xticks(rotation = 90)
    plt.ylabel('Profit Ratio')
    plt.show()
    