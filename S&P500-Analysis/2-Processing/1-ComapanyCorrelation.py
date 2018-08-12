import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pickle
import requests
import time

style.use('ggplot')

def save_sp500_tickers():
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
            else:
                print('Already Exists {}'.format(ticker))
            time.sleep(2)

def combine_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for ticker in tickers:
        df = pd.read_csv('stock_df/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        
        #df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']
        #df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']

        df.rename(columns={'Adj Close':ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

    main_df.to_csv('sp500_combine_closes.csv')

def visualize_data():
    df = pd.read_csv('sp500_combine_closes.csv')

    df.set_index('Date', inplace=True)
    df_corr = df.pct_change().corr()

    #df_corr = df.corr()
    #df_corr.to_csv('sp500corr.csv')

    data1 = df_corr.values

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap = plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    #plt.tight_layout()
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()

    

visualize_data()
            
