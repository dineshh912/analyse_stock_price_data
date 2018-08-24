import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import numpy as np
import pickle
import requests
import time
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

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
    try:
        if reload_sp500:
            tickers = save_sp500_tickers()
        else:
            with open('0-Data/sp500tickers.pickle', 'rb') as f:
                tickers = pickle.load(f)

            if not os.path.exists('stock_df'):
                os.makedirs('stock_df')

            start = dt.datetime(2016,1,1)
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
    except Exception as e:
        print(e)

def combine_data():
    with open('0-Data/sp500tickers.pickle', 'rb') as f:
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

    main_df.to_csv('stock_df/sp500_combine_closes.csv')


def visualize_data():
    df = pd.read_csv('0-Data/sp500_combine_closes.csv')

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

def process_data_labels(ticker):
    days = 5
    df = pd.read_csv('0-Data/sp500_combine_closes.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1,days+1):
        # Colum  = (new Value - old value) / new
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    columns = [c for c in args]
    requirement = 0.02
    for col in columns:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)]
                                                                 ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))
    
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X,y,df

def ml_training(ticker):
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                        y,
                                                        test_size=0.25)
    clf = neighbors.KNeighborsClassifier()
    #clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            #('knn',neighbors.KNeighborsClassifier()),
                            #('rfor',RandomForestClassifier())])
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:',Counter(predictions))


ml_training('AAPL')
    

#get_data_from_yahoo()   
#combine_data()
#visualize_data()
            
