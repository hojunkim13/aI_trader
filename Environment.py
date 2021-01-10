import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime



class Environment:
    def __init__(self,n_episode, seed, sequence_length, fee, threshold, maginot_line):
        self.start_data = datetime(2014,1,1)
        self.end_data = datetime(2019,12,31)
        self.test_start = datetime(2020,1,1)
        self.test_end =datetime.today()
        self.seed = seed
        self.sequence_length = 7
        self.fee = fee
        self.threshold = threshold
        self.n_episode = n_episode
        self.maginot_line = maginot_line
    
        


        
    def _get_code_df(self):
        df = pd.read_csv('./data/KOSPI.csv').drop(columns='NO')
        df['종목코드'] = df['종목코드'].apply(lambda x : x[3:9] + '.KS')
        return df
        
    def get_data(self):
        idx = np.random.choice(range(200))
        (code, name) = self.pool.loc[idx]
        data = pdr.get_data_yahoo(code, self.start_data, self.end_data).drop(columns = 'Adj Close')
        data = self.normalize(data)
        return data, name, code

    def normalize(self, df):
        df = df.copy()
        self.max_price = df['Close'].max()
        self.min_price = df['Close'].min()
        self.price = (self.max_price + self.min_price) / 2.
        for col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        self.unnormal = lambda x : x * (self.max_price - self.min_price) + self.min_price
        self.principal = self.unnormal(self.seed)
        return df

    #############################
    ##      For Env func       ##
    #############################
    def reset(self, episode):
        self.pool = self._get_code_df()
        self.data, self.name, self.code = self.get_data() #여기서 maxprice, minprice 만듬
        self.stock_profit = [0]
        self.inv_prin = 0
        self.stocks = 0
        self.account = self.seed
        self.idx = self.sequence_length
        self.time = self.data.index[self.idx]
        self.episode = episode
        self.total_cash = self.seed
        state = np.array(self.data.iloc[self.idx-self.sequence_length:self.idx]).reshape(-1)
    
        #print('종목: {}  코드: {} 평균가: {:.0f}원'.format(self.name, self.code, self.price))
        return state

    def step(self, order, render = False):
        done = False
        reward = 0
        buy_price = self.data['Open'][self.idx]
        price = self.data['Close'][self.idx]
        old_price = self.data['Close'][self.idx-1]
        old_account = self.account
            
        if abs(order) < self.threshold:
            order = 0
        
        #구매
            
        if order > 0 and self.account >= buy_price * order:
            if buy_price !=0:
                order = np.clip(order, 0, self.account / buy_price)
            self.account -= buy_price * order * (1+ self.fee)
            self.stocks += order
            self.inv_prin += buy_price * order * (1+ self.fee)
        #판매
        elif order < 0 and self.stocks >= order:    
            order = np.clip(abs(order), 0, self.stocks)
            price = self.data['Close'][self.idx]            
            self.account += price * order * (1- self.fee)
            self.stocks -= order
            self.inv_prin -= price * order * (1- self.fee)
            order *= -1
        self.idx += 1
        self.time = self.data.index[self.idx]

        new_state = np.array(self.data.iloc[self.idx-self.sequence_length:self.idx]).reshape(-1)
        
        
        '''
        #주식 수익률
        #현재가 / 투자금액(구매가) - 1
        if self.inv_prin != 0 and self.stocks != 0:
            stock_profit = (price * self.stocks) / self.inv_prin - 1

        else:
            stock_profit = 0
        self.stock_profit.append(stock_profit)

        #보상 = 주식 수익률
        if stock_profit < 0:
            reward = -1
        else:
            reward = 1'''
        #reward = stock_profit
        
        #현재 잔액 계산
        #오늘 평가손익 >> 어제 평가손익 이면, 보상
        
        total_cash = (self.account + self.stocks * price)
        stock_profit = (total_cash - self.total_cash) / self.total_cash
        self.stock_profit.append(stock_profit)
        
        reward += stock_profit / 10
        
            

        self.total_cash = total_cash
        #원금비율확인
        profit_ratio = (self.total_cash - self.seed) / self.seed * 100
        cash = self.unnormal(total_cash)
        
        if render:
            account = self.unnormal(self.account)
            stocks = self.unnormal(self.stocks * price)
            stock_profit = self.stock_profit[-1] * 100
            time = '[날짜: ' + str(self.time)[:10] + ']'
            print(time, f' [Account: {account:.0f} 원, Stocks: {stocks:.0f} 원] [주문: {order:.2f}] [일일 수익률: {stock_profit:.1f}]%')
        
        if self.time == self.data.index[-1] or profit_ratio < self.maginot_line:
            done = True
            print('[{}/{} 종목: {}  코드: {} 평균가: {:.0f}원'.format(self.episode+1, self.n_episode, self.name, self.code, self.price))
            print(f'###### [총 수익률: {profit_ratio:.3f}%] [원금: {self.principal:.0f}원] [자산 : {cash:.0f}원] ######')
            print('')

        
        return new_state, reward, done, profit_ratio



        