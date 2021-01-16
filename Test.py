from Agent import Agent
from Environment import Environment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
matplotlib.rcParams['axes.unicode_minus'] = False


##### params for Environment ###

n_episode = 20
sequence_length = 7
amp = 10
period = 7
clip = int(np.log10(amp))
end = datetime.today()
start = end - timedelta(days = period)

### params for Agent
path = './model/total'
render = True
state_dim = sequence_length * 10
### parms for Network
lr = 1e-4
epsilon = 0.2
gamma = 0.80
lmbda = 0.80
buffer_size = 1000
batch_size = 512
k_epochs = 10


agent = Agent(state_dim, lr = lr, epsilon = epsilon, gamma = gamma,
              lmbda = lmbda, buffer_size = buffer_size,
              batch_size= batch_size, k_epochs= k_epochs)
agent.load(path)


#same with train
def main_test():
    env = Environment(n_episode = n_episode,
                      sequence_length = sequence_length,
                      amp = amp,
                      start = start,
                      end = end
                      )

    stock_name_list = []
    score_list = []
    for e in range(n_episode):
        state = env.reset(e)
        done = False
        while not done:
            order, log_prob = agent.get_action(state)
            order = np.round(order, clip)
            state_, reward, done, profit_list = env.step(order, render)
            state = state_
        #Epi done
        stock_name_list.append(env.name)
        score_list.append(np.mean(profit_list))
    #all epi done
    count = list(filter(lambda x: x > 0, score_list))
    ratio = len(count) / len(score_list)
    average_profit = np.mean(score_list)
    print('Main Test## [평균 일일 수익률: {:.1f}, 이익 종목 비율: {:.1f}%]'.format(average_profit, ratio * 100))
    plt.scatter(stock_name_list, score_list)
    plt.xticks(rotation = 90)
    plt.ylabel('Profit Ratio [%]')
    plt.show()

#test in one item
def stock_item_test(name = None, code = None):
    env = Environment(n_episode = 1,
                      sequence_length = sequence_length,
                      amp = amp,
                      start = start,
                      end = end
                      )

    state = env.reset(0, name, code)
    done = False
    time_list = []
    while not done:
        order, log_prob = agent.get_action(state)
        order = np.round(order, clip)
        state_, reward, done, profit_list = env.step(order, render)
        time_list.append(str(env.time)[2:10])
        state = state_
    #epi done
    count = list(filter(lambda x: x < 0, profit_list))
    ratio = len(count) / len(profit_list)
    print('[손해 일수 비율: {:.1f}%]'.format(ratio * 100))
    plt.scatter(time_list, profit_list)
    plt.xticks(rotation = 90)
    plt.ylabel('Profit Ratio [%]')
    plt.title(env.name)
    plt.show()


#find items that fit good with model
def find_items():
    tested_items = pd.DataFrame(columns = ['Name', 'Code', 'Average Profit[%]',
                                           'Loss Day Ratio[%]', 'Tomorrow Prediction'])
    train_items = pd.read_csv('./data/train_item_.csv')
    num_items = len(train_items)
    env = Environment(n_episode = num_items,
                  sequence_length = sequence_length,
                  amp = amp,
                  )
    total_profit_list = []
    stock_name_list = []
    for index, (name, code) in train_items.iterrows():
        stock_name_list.append(name)
        pr_list = []
        state = env.reset(index, name, code)
        done = False
        while not done:
            order, log_prob = agent.get_action(state)
            order = np.round(order, clip)
            state_, reward, done, total_profit, pr = env.step(order, render)
            print(reward)
            state = state_
            pr_list.append(pr)
        #one items done.
        Average_profit = np.round(np.mean(pr_list), 1)
        total_profit_list.append(total_profit)
        count = list(filter(lambda x: x < 0, pr_list))
        ratio = len(count) / len(pr_list)
        ratio = np.round(ratio, 3)
        print('[평균 일일 수익률: {:.1f}, 손해 일수 비율: {:.1f}%]'.format(Average_profit, ratio * 100))        
        print('')
        order, _ = agent.get_action(state)
        order = np.round(order, clip) * amp
        #print('[매매 : {:.2f}]'.format(order))
        tested_items.loc[index] = [name, code, Average_profit, ratio * 100, order]
    #all ep done.
    tested_items.to_csv('./data/tested_item.csv', sep=',', encoding = 'utf-8-sig', index = False)
    print('평균 수익률: {:.1f}%'.format(np.mean(total_profit_list)))
    plt.scatter(stock_name_list,total_profit_list)
    plt.xticks(rotation = 90)
    plt.ylabel('Profit Ratio [%]')
    plt.show()

if __name__ == "__main__":
    main_test()
    #find_items()
    #stock_item_test()
    #stock_item_test('삼성전자', '005930.KS')