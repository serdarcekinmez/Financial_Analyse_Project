import os
import pandas as pd
import ccxt
import pandas_ta as pta
import requests
import time

# Initial setup
binance = ccxt.binance({
    'enableRateLimit': True,  # required according to the Manual
})

# Specify your directory
directory = r"C:\\Users\\serda\\OneDrive\\Bureau\\Online Education\\Python-pour-finance\\Financial Analyse Project\\binance_4h_coin_datasets"

def fetch_and_save_data(symbol, timeframe='4h'):
    print(f"Fetching {symbol} {timeframe} data...")
    data = binance.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['Time_Date'] = df['timestamp'].dt.date

    # Fetch additional data from Binance API
    start_time = int((time.time() - 100 * 24 * 3600) * 1000)
    end_time = int(time.time() * 1000)
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": timeframe,
        "startTime": start_time,
        "endTime": end_time
    }
    response = requests.get(url, params=params)
    klines = response.json()
    if len(klines) < 10:
        print(f"Skipped {symbol}: Only {len(klines)} records")
        return
    extra_df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignored'])
    extra_df['open_time'] = pd.to_datetime(extra_df['open_time'], unit='ms')

    # Merge additional data into main dataframe
    df = pd.merge(df, extra_df[['open_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']], left_on='timestamp', right_on='open_time', how='left')

    # Additional columns
    df['price'] = df['close']

    # Calculate RSI
    df['RSI_4h'] = pta.rsi(df['price'], length=14)

    # Calculate MACD
    df.ta.macd(append=True)

    # Calculate Bollinger Bands
    df.ta.bbands(length=20, append=True)

    # Calculate ADX
    df.ta.adx(append=True)

    # Calculate Stochastic RSI %k and %d values 
    df.ta.stochrsi(append=True)

    # Calculate EMA
    df['EMA_price_9_4h'] = pta.ema(df['price'], length=9)

    # Calculate ALMA
    df['ALMA_9_08_6'] = pta.alma(df['price'], window=9, offset=0.85, sigma=6)

    # Calculate Tilson T3
    df.ta.t3(append=True)

    # Calculate Fisher
    df.ta.fisher(append=True)

    # Calculate SMI Ergodic and SMI Ergodic Signal
    def smi_ergodic(data, length=20, atr_length=5, signal_length=5):
        high = data['high']
        low = data['low']
        close = data['close']

        high_low = high - low
        hl_ema = high_low.ewm(span=length).mean()
        close_change = close.diff().abs()
        cc_ema = close_change.ewm(span=length).mean()

        smi = 100 * hl_ema / (2 * cc_ema)
        smi_signal = smi.rolling(window=signal_length).mean()

        return smi, smi_signal

    df['SMI_Ergodic'], df['SMI_Ergodic_Signal'] = smi_ergodic(df)

    # Calculate Volume Weighted Moving Average (VWMA) based on the SMI Ergodic Indicator (smii)
    df['VWMA_20_SMII'] = df.ta.vwma(close='SMI_Ergodic', volume='volume', length=20)

    # Calculate hl2 and its median
    df['hl2'] = (df['high'] + df['low']) / 2
    df['median_hl2_14'] = df['hl2'].rolling(window=14).median()

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Create a filename and save the data as a CSV
    filename = f"{symbol.replace('/', '')}_binance_{timeframe}.csv"
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {symbol} {timeframe} data to {filepath}")


# List of coins
coin_symbols  = ['btc', 'near', 'grt', 'matic', 'agix', 'ankr', '1inch', 'aave', 'ada', 
                'algo', 'alice', 'ape', 'api3', 'arpa', 'ar', 'atom', 'avax', 'bake', 'bal', 
                'bat', 'blz', 'bnb', 'c98', 'cake', 'celo', 
                'chz', 'comp', 'cos', 'crv', 'dent', 
                'dock', 'doge', 'dot', 'dodo','egld', 'enj', 'eos', 
                'etc', 'eth', 'fet', 'fil', 'flm', 'flow', 'gala','icp', 'icx', 'inj',
                'iota', 'iris', 'kava', 'knc', 'lina', 'link', 'lrc', 'ltc', 
                'lunc', 'mana', 'mina', 'neo', 'ocean', 'one', 
                'phb', 'qnt', 'qtum', 'rad', 'reef', 'rndr', 'rvn', 'sand',
                'sc', 'shib', 'snx', 'sol', 'storj', 'sxp', 'theta', 
                'trb', 'trx', 'tvk', 'uma', 'uni', 'vet', 'vgx', 'wing', 'xno', 'xtz', 'xrp', 'yfi', 'zil']

coin_symbols = [(coin + 'usdt').upper() for coin in coin_symbols]

# Fetch and save data for each coin
for coin in coin_symbols:
    fetch_and_save_data(coin)
    time.sleep(1)
