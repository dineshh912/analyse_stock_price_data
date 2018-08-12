import bs4 as bs
import pickle
import requests

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

save_sp500_tickers()

