from Agent import Agent
from Environment import Environment
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
matplotlib.rcParams['axes.unicode_minus'] = False



switch = 'find_item'
##### params for Environment ###
n_episode = 20
sequence_length = 7
amp = 10
period = 8
end = datetime.today()
start = end - timedelta(days = period)

### params for Agent
path = './model/total2'
render = False
filter_item = True
state_dim = sequence_length * 10
### parms for Network
lr = 1e-5
epsilon = 0.2
gamma = 0.5
lmbda = 0.5
buffer_size = 5000
batch_size = 512
k_epochs = 30


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
            state_, reward, done, profit_list = env.step(order, render)
            state = state_
        #Epi done
        stock_name_list.append(env.name)
        score_list.append(np.mean(profit_list))
    #all epi done
    count = list(filter(lambda x: x > 0, score_list))
    ratio = len(count) / len(score_list)
    average_profit = np.round(np.mean(score_list), 3)
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
        state_, reward, done, profit_list = env.step(order, render)
        time_list.append(str(env.time)[2:10])
        state = state_
    #epi done
    count = list(filter(lambda x: x > 0, profit_list))
    ratio = len(count) / len(profit_list)
    print('##Item Test [Successful pred ratio]: {:.1f}%]'.format(ratio * 100))
    plt.scatter(time_list, profit_list)
    plt.xticks(rotation = 90)
    plt.ylabel('Profit Ratio [%]')
    plt.title(env.name)
    plt.show()


#find items that fit good with model
def find_items():
    all_items = pd.read_csv('./data/KOSPI.csv', index_col='NO')
    all_items['종목코드'] = all_items['종목코드'].apply(lambda x : x[3:9] + '.KS')
    num_items = len(all_items)
    env = Environment(n_episode = num_items,
                      sequence_length = sequence_length,
                      amp = amp,
                      start = start,
                      end = end,
                      )
    info_list = []
    avr_score_list = []
            
    for index, (code, name) in all_items.iterrows():
        state = env.reset(index-1, name, code)
        done = False
        while not done:
            order, log_prob = agent.get_action(state)
            state_, reward, done, pr_list = env.step(order, render)
            state = state_
        #one items done.
        average_profit =np.mean(pr_list)
        avr_score_list.append(average_profit)
        count = list(filter(lambda x: x > 0, pr_list))
        ratio = len(count) / len(pr_list)
        ratio = np.round(ratio, 3)
        print('[평균 일일 수익률: {:.1f}, 예측 정확도: {:.1f}%]'.format(average_profit, ratio * 100))        
        #print('')
        order, _ = agent.get_action(state)
        order = np.round(order, 1) * amp
        info_list.append((env.name, env.code, average_profit, ratio * 100, order))
    #all ep done.
    columns = ['Name', 'Code', 'Average Profit[%]', 'Pred Accuracy[%]', 'Tomorrow Pred']
    filtered_items = pd.DataFrame.from_records(info_list, columns = columns)
    if filter_item:
        filtered_items.to_csv('./data/items.csv', sep=',', encoding = 'utf-8-sig', index = False)
    print('평균 수익률: {:.1f}%'.format(np.mean(avr_score_list)))
    plt.scatter(range(len(avr_score_list)), avr_score_list)
    plt.ylabel('Profit Ratio [%]')
    plt.show()


def run(switch):
    if switch =='main_test':
        main_test()
    elif switch =='find_item':
        find_items()
    elif switch =='item_test':
        stock_item_test()


if __name__ == "__main__":
    run(switch)