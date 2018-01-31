from __future__ import division
import pandas as pd
import numpy as np
from scipy import stats
from pandas_datareader.data import Options
from scipy.optimize import brentq
from math import log, sqrt, exp
from pandas_datareader.data import Options
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError

import time


def main():
    ticker_list=pd.read_csv('116/NasdaqStockList.csv')

    available_list=[]
    attention_list=[]

    aggregate_df=pd.DataFrame()
    for ticker in ticker_list.ticker:
        try:
            print ticker

            available_list.append(ticker)

            t=OptionAggregate(ticker)
            t.Option_get()
            aggregate_df=pd.concat([aggregate_df,t.dataframe])

            time.sleep(3)
        except RemoteDataError:
            continue
        except ConnectionError:
            time.sleep(5)
            print ticker+'!!!!!!!!!!'
            attention_list.append(ticker)
            continue


def expiration (df):
    df['Quote_Time']=pd.to_datetime(df['Quote_Time'])
    df['Expiry'] = pd.to_datetime(df['Expiry'])
    Expiration =( (360 *(df['Expiry'].dt.year -df['Quote_Time'].dt.year)
         + 30*(df['Expiry'].dt.month-df['Quote_Time'].dt.month))
    + (df['Expiry'].dt.day-df['Quote_Time'].dt.day))/360
    df = df.assign(Expiration = Expiration.values)
    return df


def raw_get(ticker_name):
    aapl = Options(ticker_name, 'yahoo')
    data = aapl.get_all_data()
    data.reset_index(inplace=True)
    data.drop('JSON',axis=1,inplace=True)
    return data


def clean(data):
    #europe
    data = expiration(data)

    columns = ['IsNonstandard','Underlying','Symbol','Chg','PctChg','Open_Int']
    data = data.drop(columns, axis=1)

    return data


def _getVolatility(t,history_data):
    '''input: ticker name & last trade date & historical_period
    output: historical volatility'''
    # set end date: last trade date
    # set start date: use time-to-maturity get historical time period
    end=t['Last_Trade_Date'].date()
    days_between = t['Expiry'].date()-t['Last_Trade_Date'].date()
    start = end - days_between

    ticker_dataframe = history_data[start:end]

    quotes = ticker_dataframe['Close']
    logreturns = np.log(quotes / quotes.shift(1))
    volatility = np.sqrt(252 * logreturns.var())

    return volatility


def HV_calculate(data,ticker):
    """

    :param data: cleaned options data for one ticker
    :param ticker: ticker name
    :return:data with volatility column

    """
    min_date=data.apply(lambda r: r['Last_Trade_Date'].date() - (r['Expiry'].date() - r['Last_Trade_Date'].date()), axis=1).min()
    max_date=data['Last_Trade_Date'].max().date()
    history_data = web.DataReader(ticker, data_source='yahoo', start=min_date, end=max_date)

    HV_series=data.apply(lambda r:_getVolatility(r,history_data),axis=1)
    data['volatility']=HV_series
    return data

class OptionAggregate():
    def __init__(self,ticker):
        self.ticker=ticker
    def Option_get(self):
        self.clean_data=clean(raw_get(self.ticker))
        self.dataframe=HV_calculate(self.clean_data,self.ticker)

if __name__ == '__main__':
    main()

