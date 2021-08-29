import os
import pandas as pd
import numpy as np

import talib.abstract as ta
from finta import TA as fta
import indicators as qtpylib

from scipy.signal import savgol_filter

# Read json OHLC pair data
def get_json_data(pair, timeframe, indicators=False, dropna=False,
                  labels=False, label_window_size=9, label_col_name='close',
                  labels_trend=False):

    df = pd.read_json(f'/home/falonso/tmp/mltrader/data/{pair}-{timeframe}.json', orient='values')
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    df = df.astype(dtype={'open': 'float', 'high': 'float',
                          'low': 'float', 'close': 'float', 'volume': 'float'})

    df['date'] = pd.to_datetime(df['date'],
                                unit='ms',
                                utc=True,
                                infer_datetime_format=True)

    if indicators:
        add_indicators(df)

    if labels:
        add_labels(df, label_window_size, label_col_name)

    if labels_trend:
        add_labels_trend(df)

    if dropna:
        df = df.dropna()

    return df


indicators_wo_normalization = [
    'updn',
    'STX_7_4', 'STX_10_3', 'STX_14_2',
    ]

def add_indicators(df):

    # Mid-price
    df['mid1'] = 0.5 * (df['open'] + df['close'])
    df['mid2'] = 0.5 * (df['low']  + df['high'])

    # Candle direction
    df['updn'] = np.where(df['close'] > df['open'], 1, -1)

    #
    df['spread'] = df['high'] - df['low']

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
    for p in (5, 6, 12, 20, 21, 50, 55, 100, 110, 250):
        df[f'TEMA-{p}'] = ta.TEMA(df, timeperiod=p)
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
    df['ADX-14'] = ta.ADX(df, period=14)

    df['ADOSC'] = ta.ADOSC(df)

    # Parabolic SAR
    df['SAR'] = ta.SAR(df)

    # Bollinger Bands because obviously
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=1)
    df['bb_low_20_1'] = bollinger['lower']
    df['bb_mid_20_1'] = bollinger['mid']
    df['bb_upp_20_1'] = bollinger['upper']
    df["bb_per_20_1"] = ((df["close"] - df["bb_low_20_1"]) / (df["bb_upp_20_1"] - df["bb_low_20_1"]))
    df["bb_wid_20_1"] = ((df["bb_upp_20_1"] - df["bb_low_20_1"]) / df["bb_mid_20_1"])

    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df['bb_low_20_2'] = bollinger['lower']
    df['bb_mid_20_2'] = bollinger['mid']
    df['bb_upp_20_2'] = bollinger['upper']
    df["bb_per_20_2"] = ((df["close"] - df["bb_low_20_2"]) / (df["bb_upp_20_2"] - df["bb_low_20_2"]))
    df["bb_wid_20_2"] = ((df["bb_upp_20_2"] - df["bb_low_20_2"]) / df["bb_mid_20_2"])

    # Bollinger Bands - Weighted (EMA based instead of SMA)
    weighted_bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df["wbb_upp"] = weighted_bollinger["upper"]
    df["wbb_low"] = weighted_bollinger["lower"]
    df["wbb_mid"] = weighted_bollinger["mid"]
    df["wbb_per"] = ((df["close"] - df["wbb_low"]) / (df["wbb_upp"] - df["wbb_low"]))
    df["wbb_wid"] = ((df["wbb_upp"] - df["wbb_low"]) / df["wbb_mid"])

    # # Chart type
    # # ------------------------------------
    # # Heikin Ashi Strategy
    heikinashi = qtpylib.heikinashi(df)
    df['ha_open'] = heikinashi['open']
    df['ha_close'] = heikinashi['close']
    df['ha_high'] = heikinashi['high']
    df['ha_low'] = heikinashi['low']

    # Cycle Indicator
    # ------------------------------------
    # Hilbert Transform Indicator - SineWave
    hilbert = ta.HT_SINE(df)
    df['htsine'] = hilbert['sine']
    df['htleadsine'] = hilbert['leadsine']

    # Volume Flow Indicator (MFI) for volume based on the direction of price movement
    df['VFI'] = fta.VFI(df, period=14)

    dmi = fta.DMI(df, period=14)
    df['dmi_plus'] = dmi['DI+']
    df['dmi_minus'] = dmi['DI-']

    add_supertrend(df, 7, 4)
    add_supertrend(df, 10, 3)
    add_supertrend(df, 14, 2)

    # # Pattern Recognition - Bullish/Bearish candlestick patterns
    # # ------------------------------------
    # # # Three Line Strike: values [0, -100, 100]
    # df['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(df)
    # # # Spinning Top: values [0, -100, 100]
    # df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df) # values [0, -100, 100]
    # # # Engulfing: values [0, -100, 100]
    # df['CDLENGULFING'] = ta.CDLENGULFING(df) # values [0, -100, 100]
    # # # Harami: values [0, -100, 100]
    # df['CDLHARAMI'] = ta.CDLHARAMI(df) # values [0, -100, 100]
    # # # Three Outside Up/Down: values [0, -100, 100]
    # df['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(df) # values [0, -100, 100]
    # # # Three Inside Up/Down: values [0, -100, 100]
    # df['CDL3INSIDE'] = ta.CDL3INSIDE(df) # values [0, -100, 100]

    # # Pattern Recognition - Bullish candlestick patterns
    # # ------------------------------------
    # # # Hammer: values [0, 100]
    # df['CDLHAMMER'] = ta.CDLHAMMER(df)
    # # # Inverted Hammer: values [0, 100]
    # df['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df)
    # # # Dragonfly Doji: values [0, 100]
    # df['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df)
    # # # Piercing Line: values [0, 100]
    # df['CDLPIERCING'] = ta.CDLPIERCING(df) # values [0, 100]
    # # # Morningstar: values [0, 100]
    # df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df) # values [0, 100]
    # # # Three White Soldiers: values [0, 100]
    # df['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df) # values [0, 100]

    # # Pattern Recognition - Bearish candlestick patterns
    # # ------------------------------------
    # # # Hanging Man: values [0, 100]
    # df['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df)
    # # # Shooting Star: values [0, 100]
    # df['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df)
    # # # Gravestone Doji: values [0, 100]
    # df['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df)
    # # # Dark Cloud Cover: values [0, 100]
    # df['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df)
    # # # Evening Doji Star: values [0, 100]
    # df['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(df)
    # # # Evening Star: values [0, 100]
    # df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df)


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


def add_labels_trend(df):

    smooth_window = 9 # 2.25 hs
    w_trend = 0.01

    # smooth close price
    df['sclose'] = savgol_filter(df['close'], smooth_window, 3, deriv=0)

    N = len(df)

    labels = np.zeros(N)

    x_0 = df.iloc[0]['sclose']

    x_l, x_h = x_0, x_0
    lt, ht = 0, 0
    cid = 0
    fp, fp_n = x_0, 0

    for i in range(N):
        x_i = df.iloc[i]['sclose']
        if x_i > fp + x_0 * w_trend:
            x_h, ht, fp_n, cid = x_i, i, i, 1
            break

        if x_i < fp - x_0 * w_trend:
            x_l, lt, fp_n, cid = x_i, i, i, -1
            break

    for i in range(fp_n+1, N):

        x_i = df.iloc[i]['sclose']

        if cid > 0:
            if x_i > x_h:
                x_h, ht = x_i, i
            if x_i < x_h - x_h * w_trend and lt <= ht:
                for j in range(lt, ht+1):
                    labels[j] = 1
                x_l, lt, cid = x_i, i, -1

        elif cid < 0:
            if x_i < x_l:
                x_l, lt = x_i, i-1
            if x_i > x_l + x_l * w_trend and ht <= lt:
                for j in range(ht, lt):
                    labels[j] = 0
                x_h, ht, cid = x_i, i, 1


    df['label'] = labels


def add_supertrend(df, period=14, multiplier=4):

    st = f'ST_{period}_{multiplier}'
    stx = f'STX_{period}_{multiplier}'

    df['TR'] = ta.TRANGE(df)
    df[f'ATR-{period}'] = ta.SMA(df['TR'], period)

    # Compute basic upper and lower bands
    df['avg'] = 0.5 * (df['high'] + df['low'])
    df['basic_ub'] = df['avg'] + multiplier * df[f'ATR-{period}']
    df['basic_lb'] = df['avg'] - multiplier * df[f'ATR-{period}']

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] >  df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
            df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] <  df['final_lb'].iat[i] else 0.00

    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), -1,  1), 0)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    return


def split_train_val(df, frac=0.8):

    n = len(df)

    df_train = df[:int(n*frac)]
    df_valid = df[int(n*frac):n]

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    return df_train, df_valid
