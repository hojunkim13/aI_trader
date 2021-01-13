import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime, timedelta
from wcwidth import wcswidth


class Environment:
    def __init__(self,n_episode, sequence_length, amp,
                 start = datetime(2010,1,1), end = datetime(2020,1,1)):
        self.pool = self._get_code_df()
        self.start = self.check_time(start)
        self.end = self.check_time(end)
        self.sequence_length = sequence_length
        self.amp = amp
        self.n_episode = n_episode
        self.choice_stock = np.random.choice(range(200))


    def _get_code_df(self):
        df = pd.read_csv('./data/KOSPI.csv').drop(columns='NO')
        df['종목코드'] = df['종목코드'].apply(lambda x : x[3:9] + '.KS')
        return df
        
    def get_data(self, name = None, code = None):
        if name == None:
            choice_stock = self.choice_stock % 200
            self.choice_stock += 1
            (code, name) = self.pool.loc[choice_stock]
        start = datetime(2010,1,1) + timedelta(days= -20)
        end = self.check_time(datetime.today())
        
        self.total_data = pdr.get_data_yahoo(code, start, end).drop(columns = 'Adj Close')
        self.max_price = self.total_data['Close'].max()
        self.min_price = self.total_data['Close'].min()

        period = (self.end - self.start).days + self.sequence_length + 1
        target_period_data = pdr.get_data_yahoo(code, self.start + timedelta(days=-20), self.end).drop(columns = 'Adj Close')
        return target_period_data[-period:], name, code

    def normalize(self, df):
        df = df.copy()
        local_max = df['Close'].min()
        local_min = df['Close'].min()
        for col in df.columns:
            g_max = self.total_data[col].max()
            g_min = self.total_data[col].min()
            l_min = df[col].min()
            l_max = df[col].max()
            assert g_max >= l_max, (self.name+', normalize error occured,  ' + col)
            assert g_min <= l_min, (self.name+', normalize error occured,  ' + col)
            df[col] = (df[col] - g_min) / (g_max - g_min)
            #H L O C V
            if col == 'Open':
                self.unnorm_o = [(g_max - g_min), g_min]
            elif col == 'Close':
                self.unnorm_c = [(g_max - g_min), g_min]

        return df
    
    def check_time(self, date):
        if date.date()==datetime.now().date() and datetime.now().hour + datetime.now().minute/60 < 15.5:
            date = datetime.today() - timedelta(days=1)
        return date

    #############################
    ##      For Env func       ##
    #############################
    def reset(self, episode, name = None, code = None):
        self.profit_list = []
        data, self.name, self.code = self.get_data(name, code)
        self.data = self.normalize(data)
        self.idx = self.sequence_length
        self.time = self.data.index[self.idx]
        self.episode = episode
        state = np.array(self.data.iloc[self.idx-self.sequence_length:self.idx]).reshape(-1)
        return state

    def step(self, order, render = False):
        #H L O C V
        buy = self.data['Open'][self.idx] * self.unnorm_o[0] + self.unnorm_o[1]
        sell = self.data['Close'][self.idx] * self.unnorm_c[0] + self.unnorm_c[1]
        
        done = False
        order = order * self.amp
        #계산
        #in case of order < 0, (pred means sell & price will go down)
        #if price difference is plus, our income will be minus,
        #if price difference is minus, our income will be plus.
        d_price = (sell - buy)
        #reward calc
        income = d_price * order

        profit = d_price / buy * np.sign(order)

        self.profit_list.append(profit*100)
        #todo sign >> clip (price clipping)
        if np.sign(d_price) < 0:
            reward = np.sign(d_price) * (order) / self.amp 
            reward = np.clip(reward, 0, 1)
        else:
            reward = np.sign(d_price) * (order) / self.amp * 0.5
            reward = np.clip(reward, 0, 1)

        if render:
            sign = f'{np.sign(d_price):+1.0f}'[0]
            time = '[날짜: ' + str(self.time)[:10] + ']'
            print(time, f' [주문(today): {order:.0f}] [가격변동: {sign}] [일일 수익률: {(profit * 100):.1f}%] [수익: {income:.0f} 원]')
        
        self.idx += 1
        new_state = np.array(self.data.iloc[self.idx-self.sequence_length:self.idx]).reshape(-1)
        self.time = self.data.index[self.idx]

        if self.time == self.data.index[-1]:
            done = True
            average_profit = np.mean(self.profit_list)
            length = len(self.name)
            pad = ' ' * (13 - length) * 2 
            print(f'[{self.episode+1}/{self.n_episode}] 종목: {self.name}{pad}', end='')
            print(f'코드: {self.code}  평균 일일 수익률: {average_profit:+.2f}%')
            return new_state, reward, done, self.profit_list
        
        return new_state, reward, done, None