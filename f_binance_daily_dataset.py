import pandas as pd
import requests
import time
import os
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as pta


coin_symbols = [ 
                'btc', 'near', 'grt', 'matic', 'agix', 'ankr', '1inch', 'aave', 'ada', 'algo', 'alice', 
                'ape', 'api3', 'arpa', 'ar', 'atom', 'avax', 'bake', 'bal', 'bat', 'blz', 'bnb', 
                'c98', 'cake', 'celo', 'chz','comp', 'cos', 'crv', 'dent', 'dock', 'doge', 'dot', 'dodo',
                'egld', 'enj', 'eos', 'etc', 'eth', 'fet', 'fil', 'flm', 'flow', 'gala','icp', 'icx', 'inj',
                'iota', 'iris', 'kava', 'knc', 'lina', 'link', 'lrc', 'ltc', 'lunc', 'mana', 'mina','neo', 'ocean', 'one', 
                'phb', 'qnt', 'qtum', 'rad', 'reef', 'rndr', 'rvn', 'sand','sc', 'shib', 'snx', 'sol', 'storj', 'sxp', 
                'theta', 'trb', 'trx', 'tvk', 'uma', 'uni', 'vet', 'vgx','wing', 'xno', 'xtz', 'xrp', 'yfi', 'zil']

symbols = [symbol.upper() + 'USDT' for symbol in coin_symbols]

interval = "1d"
end_time = int(time.time() * 1000)

scaler = MinMaxScaler(feature_range=(0, 1))

data = []
for symbol in symbols:
    start_time = int((time.time() - 300 * 24 * 3600) * 1000)
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time
    }
    response = requests.get(url, params=params)
    klines = response.json()
    
    df = pd.DataFrame(klines, columns=['open_time','open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored'])
    df['symbol'] = symbol
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time']= pd.to_datetime(df['close_time'], unit='ms')
    df['Time_Date'] = df['close_time'].dt.date
    
    df = df[['symbol', 'open_time', 'close_time','Time_Date','open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])

    df['RSI_daily'] = pta.rsi(df['close'], length=14)


    df['scaled_price'] = scaler.fit_transform(df['close'].values.reshape(-1,1))
    df['scaled_price'].fillna(df['scaled_price'].mean(), inplace=True)
    
    data.append(df)
    
    # Save DataFrame to a CSV file
    folder_path = "C:/Users/serda/OneDrive/Bureau/Online Education/Python-pour-finance/Financial Analyse Project/binance_daily"
    filename = os.path.join(folder_path, f"{symbol}_binance_1d.csv")
    
    df.to_csv(filename, index=False)
    
    time.sleep(1)  # Add a 1-second delay between requests

# data_processing.py
def scaled_daily_price():
    return df[['close_time', 'scaled_price']]


