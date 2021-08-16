import pandas as pd
import numpy as np
import talib.abstract as ta

# Read json OHLC pair data
def get_json_data(path, indicators=False, dropna=False, labels=False, label_window_size=9, label_col_name='close'):

    df = pd.read_json(path, orient='values')
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    df = df.astype(dtype={'open': 'float', 'high': 'float',
                          'low': 'float', 'close': 'float', 'volume': 'float'})

    df['date'] = pd.to_datetime(df['date'],
                                unit='ms',
                                utc=True,
                                infer_datetime_format=True)

    if indicators:
        add_indicators(df)

    if dropna:
        df = df.dropna()

    if labels:
        add_labels(df, label_window_size, label_col_name)

    return df


def add_indicators(df):

    # Mid-price
    df['mid1'] = 0.5 * (df['open'] + df['close'])
    df['mid2'] = 0.5 * (df['low']  + df['high'])

    # Candle direction
    df['updn'] = np.where(df['close'] > df['open'], 1, 0)

    df['RSI'] = ta.RSI(df)

    # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
    rsi = 0.1 * (df['RSI'] - 50)
    df['Fisher_RSI'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

    # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
    df['Fisher_RSI_norma'] = 50 * (df['Fisher_RSI'] + 1)

    macd = ta.MACD(df)
    df['MACD'] = macd['macd']
    df['MACDSIGNAL'] = macd['macdsignal']
    df['MACDHIST'] = macd['macdhist']

    # 'MACD-0',               # Moving Average Convergence/Divergence
    #     'MACD-1',               # Moving Average Convergence/Divergence
    #     'MACD-2',               # Moving Average Convergence/Divergence
    #     'MACDEXT-0',            # MACD with controllable MA type
    #     'MACDEXT-1',            # MACD with controllable MA type
    #     'MACDEXT-2',            # MACD with controllable MA type
    #     'MACDFIX-0',            # Moving Average Convergence/Divergence Fix 12/26
    #     'MACDFIX-1',            # Moving Average Convergence/Divergence Fix 12/26
    #     'MACDFIX-2',            # Moving Average Convergence/Divergence Fix 12/26


    df['MFI'] = ta.MFI(df)
    df['ROC'] = ta.ROC(df)

    # EMA - Exponential Moving Average
    # SMA - Simple Moving Average
    for p in (5, 6, 12, 21, 50, 55, 100, 110, 250):
        df[f'EMA-{p}'] = ta.EMA(df, timeperiod=p)
        df[f'SMA-{p}'] = ta.SMA(df, timeperiod=p)

    # Plus Directional Indicator / Movement
    df['PLUS_DM'] = ta.PLUS_DM(df)
    df['PLUS_DI'] = ta.PLUS_DI(df)

    # Minus Directional Indicator / Movement
    df['MINUS_DM'] = ta.MINUS_DM(df)
    df['MINUS_DI'] = ta.MINUS_DI(df)

    # Aroon, Aroon Oscillator
    aroon = ta.AROON(df)
    df['AROON_UP'] = aroon['aroonup']
    df['AROON_DN'] = aroon['aroondown']
    df['AROON_SC'] = ta.AROONOSC(df)

    # Stochastic Slow
    stoch = ta.STOCH(df)
    df['SLOWD'] = stoch['slowd']
    df['SLOWK'] = stoch['slowk']

    # Stochastic Fast
    stoch_fast = ta.STOCHF(df)
    df['FASTD'] = stoch_fast['fastd']
    df['FASTK'] = stoch_fast['fastk']

    df['ADX'] = ta.ADX(df)
    df['ADOSC'] = ta.ADOSC(df)

    # Parabolic SAR
    df['SAR'] = ta.SAR(df)

    # TEMA - Triple Exponential Moving Average
    df['TEMA'] = ta.TEMA(df, timeperiod=9)


def add_labels(df, window_size, col_name):

    total_rows = len(df)

    labels_b = np.zeros(total_rows)
    labels_s = np.zeros(total_rows)
    labels_h = np.zeros(total_rows)

    for row in range(window_size-1, total_rows):

        window_beg = row - (window_size - 1)
        window_end = row
        window_mid = int(0.5 * (window_beg + window_end))

        prices = df.iloc[window_beg:window_end+1][col_name].to_numpy()

        min_idx = window_beg + np.argmin(prices)
        max_idx = window_beg + np.argmax(prices)
    
        if max_idx == window_mid:
            labels_s[window_mid] = 1
        elif min_idx == window_mid:
            labels_b[window_mid] = 1
        else:
            labels_h[window_mid] = 1

    df['label_s'] = labels_s.astype(int)
    df['label_b'] = labels_b.astype(int)
    df['label_h'] = labels_h.astype(int)

