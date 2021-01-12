import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime



class Environment:
    def __init__(self,n_episode, seed, sequence_length, fee, amp, maginot_line,
                 start = datetime(2015,1,1), end = datetime(2020,1,1)):
        self.pool = self._get_code_df()
        self.start = start
        self.end = end
        self.seed = seed
        self.sequence_length = sequence_length
        self.fee = fee
        self.amp = amp
        self.n_episode = n_episode
        self.maginot_line = maginot_line
        self.threshold = 0.0
    
        
    def _get_code_df(self):
        df = pd.read_csv('./data/KOSPI.csv').drop(columns='NO')
        df['종목코드'] = df['종목코드'].apply(lambda x : x[3:9] + '.KS')
        return df
        
    def get_data(self):
        choice_stock = np.random.choice(range(200))
        (code, name) = self.pool.loc[choice_stock]

        start = datetime(2015,1,1)
        end = datetime.today()
        self.total_data = pdr.get_data_yahoo(code, start, end).drop(columns = 'Adj Close')
        self.max_price = self.total_data['Close'].max()
        self.min_price = self.total_data['Close'].min()
        self.price = (self.max_price + self.min_price) / 2.

        target_period_data = pdr.get_data_yahoo(code, self.start, self.end).drop(columns = 'Adj Close')
        target_period_data = self.normalize(target_period_data)
        return target_period_data, name, code

    def normalize(self, df):
        df = df.copy()
        local_max = df['Close'].min()
        local_min = df['Close'].min()
        for col in df.columns:
            g_max = self.total_data[col].max()
            g_min = self.total_data[col].min()
            l_min = df[col].min()
            l_max = df[col].max()
            assert g_max >= l_max, 'normalize error occured.'
            assert g_min <= l_min, 'normalize error occured.'
            df[col] = (df[col] - g_min) / (g_max - g_min)
        self.unnormal = lambda x : x * (self.max_price - self.min_price) + self.min_price
        self.principal = self.unnormal(self.seed)
        return df

    #############################
    ##      For Env func       ##
    #############################
    def reset(self, episode):
        self.data, self.name, self.code = self.get_data()
        self.account = self.seed
        self.idx = self.sequence_length
        self.time = self.data.index[self.idx]
        self.episode = episode
        state = np.array(self.data.iloc[self.idx-self.sequence_length:self.idx]).reshape(-1)
    
        return state

    def step(self, order, render = False):
        buy_price = self.data['Open'][self.idx]
        price = self.data['Close'][self.idx]

        done = False
        reward = 0
        order = order * self.amp
        if buy_price !=0:
            order = np.clip(order, 0, self.account / buy_price)
        
        #계산
        if order > self.threshold * self.amp:
            buy = buy_price * order * (1 + self.fee)
            sell = price * order * (1 - self.fee)
            income = sell - buy
            account = self.account + income
        else:
            account = self.account
            income = 0
        self.idx += 1
        self.time = self.data.index[self.idx]
        new_state = np.array(self.data.iloc[self.idx-self.sequence_length:self.idx]).reshape(-1)
        
        
        #오늘 평가손익 >> 어제 평가손익 이면, 보상
        #reward calc
        profit = (account - self.account) / self.account
        reward += profit
        self.account = account
        
        #원금비율확인
        profit_ratio = (self.account - self.seed) / self.seed * 100
        cash = self.unnormal(self.account)
        income_cash = cash * profit
        
        if render:
            time = '[날짜: ' + str(self.time)[:10] + ']'
            print(time, f' [자산: {cash:.0f}원] [주문(today): {order:.2f}] [일일 수익률: {(profit*100):.1f}%] [수익: {income_cash:.0f} 원]')
        
        if self.time == self.data.index[-1] or profit_ratio <= self.maginot_line:
            done = True
            print('[{}/{}] 종목: {}  코드: {}  평균가: {:.0f}원'.format(self.episode+1, self.n_episode, self.name, self.code, self.price))
            print(f'###### [총 수익률: {profit_ratio:.1f}%] [원금: {self.principal:.0f}원] [자산 : {cash:.0f}원] ######')
            if render:
                print('')
        return new_state, reward, done, profit_ratio, profit*100

if __name__ == '__main__':
    env = Environment(10,10,7,0.015,10,-100,)
    state = env.reset()
    


        