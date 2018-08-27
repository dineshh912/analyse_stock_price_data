import datetime as dt
import os, pickle, requests, time
import pandas as pd
import pandas_datareader.data as web


# Top 500 Companies based on Market Capitalisation as on March 31, 2018

def Save_NSE_tickers():
    df = pd.read_csv(open('data/TOP500_BRR_31032018.csv','r'))
    tickers = df['Symbol'].values.tolist()
    with open('data/NSE.pickle', 'wb') as f:
        pickle.dump(tickers,f)

    return tickers

def get_data_from_yahoo(reload_NSE = False):
    if reload_NSE:
        tickers = Save_NSE_tickers()
    else:
        with open('data/NSE.pickle', 'rb') as f:
            tickers = pickle.load(f)

        if not os.path.exists('data/stock_df'):
            os.makedirs('data/stock_df')
                
        start = dt.datetime(2016,1,1)
        end  = dt.datetime.now()
        for ticker in tickers:
            if not os.path.exists('data/stock_df/{}.csv'.format(ticker)):
                NSI = ticker + '.NS'
                
                df = web.DataReader(NSI, 'yahoo', start, end)
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df.to_csv('data/stock_df/{}.csv'.format(ticker))
                #print('{} CSV has been created').format(NSI)
            else:
                print('Already Exists {}'.format(ticker))
            time.sleep(2)
        
#Save_NSE_tickers()
get_data_from_yahoo()
