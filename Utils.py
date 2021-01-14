import pandas as pd


def SMA(data, period=30, column='Close'):
    return data[column].rolling(window=period).mean()


def EMA(data, period=20, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()


def MACD(data, period_long=26, period_short=12, period_signal=9, column='Close'):
    ShortEMA = EMA(data, period_short, column=column)
    LongEMA = EMA(data, period_long, column=column)
    data['MACD'] = ShortEMA - LongEMA
    data['Signal_Line'] = EMA(data, period_signal, column='MACD')
    return data


def RSI(data, period = 14, column = 'Close'):
    delta = data[column].diff(1)
    delta = delta.dropna() # or delta[1:]

    up =  delta.copy()  # delta 값 복사
    down = delta.copy() # delta 값 복사
    up[up < 0] = 0 
    down[down > 0] = 0 

    AVG_Gain = up.rolling(window=period).mean()
    AVG_Loss = abs(down.rolling(window=period).mean())
    RS = AVG_Gain / AVG_Loss

    RSI = 100.0 - (100.0/ (1.0 + RS))
    data['RSI'] = RSI
    return data
