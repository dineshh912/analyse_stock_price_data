import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
import time

# Saving S&P 500 company tickers sympol 
def save_sp500_tickers():
    # Getting sp500 company tickers from wikipedia or any other resources using BS and save as a pickel file
    '''resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies') 
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table',{'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    '''
    resp = requests.get('https://www.slickcharts.com/sp500')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'table table-hover'})
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[2].text
        tickers.append(ticker)
        
    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers,f)

    return tickers

# Getting Values of the S&P 500 companies
def get_data_from_yahoo(reload_sp500 = False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

        if not os.path.exists('stock_df'):
            os.makedirs('stock_df')

        start = dt.datetime(2000,1,1)
        end  = dt.datetime.now()

        for ticker in tickers:
            if not os.path.exists('stock_df/{}.csv'.format(ticker)):
                df = web.DataReader(ticker, 'yahoo', start, end)
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df.to_csv('stock_df/{}.csv'.format(ticker))
                print(ticker)
            else:
                print('Already Exists {}'.format(ticker))
            #time.sleep(2)

get_data_from_yahoo()
