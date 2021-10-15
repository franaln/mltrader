import os
import pandas as pd
import numpy as np
import itertools

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import talib.abstract as ta
from finta import TA as fta
import indicators as qtpylib

from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.linear_model import LinearRegression

from labels import add_labels, add_labels_trend

# Read json OHLC pair data
def get_json_data(pair=None, timeframe=None, path=None, indicators=False, dropna=False,
                  labels=False, label_window_size=9, label_col_name='close',
                  labels_trend=False,
                  last_n=None,
                  start=None, end=None):

    # if path is None:
    path = f'data/{pair}-{timeframe}.json'

    df = pd.read_json(path, orient='values')
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    df = df.astype(dtype={'open': 'float', 'high': 'float',
                          'low': 'float', 'close': 'float', 'volume': 'float'})

    df['date'] = pd.to_datetime(df['date'],
                                unit='ms',
                                utc=True,
                                infer_datetime_format=True)

    df = df.set_index(['date'])

    if indicators:
        df = add_indicators(df)

    if labels:
        add_labels(df, label_window_size, label_col_name)

    if labels_trend:
        add_labels_trend(df)

    if dropna:
        df = df.dropna()

    if start is not None and end is not None:
        df = df.loc[start:end]

    if last_n is not None:
        df = df.drop(df.head(len(df)-last_n).index)

    # save
    df.to_json(f'data/{pair}-{timeframe}_preprocessed.json')

    return df


def gradient_past(a):
    grad = []
    for i in range(len(a)):
        if i < 2:
            grad.append(0)
        else:
            grad.append(np.gradient(a[:i])[-1])

    return np.array(grad)


def slope(series):

    y = series.values.reshape(-1,1)
    x = np.array(range(1, series.shape[0] + 1)).reshape(-1,1)

    model = LinearRegression()
    model.fit(x, y)

    slope = model.coef_

    return slope


def add_indicators(df):

    # SMA/EMA/TEMA
    for p in (3, 5, 6, 12, 15, 20, 21, 50, 55, 100, 110, 250):
        df[f'SMA-{p}']  = ta.SMA(df, timeperiod=p)
        df[f'EMA-{p}']  = ta.EMA(df, timeperiod=p)
        df[f'TEMA-{p}'] = ta.TEMA(df, timeperiod=p)

    # EMA gradient without using future date
    df['gEMA-5']   = gradient_past(df['EMA-5'])
    df['gEMA-21']  = gradient_past(df['EMA-21'])
    df['gEMA-55']  = gradient_past(df['EMA-55'])
    df['gEMA-110'] = gradient_past(df['EMA-110'])

    # diff with EMA-X
    df['dEMA-50']  = (df['close'] - df['EMA-50'])  ##/ df['mid_lh']
    df['dEMA-100'] = (df['close'] - df['EMA-100']) ##/ df['mid_lh']
    df['dEMA-250'] = (df['close'] - df['EMA-250']) ##/ df['mid_lh']

    # EMA ratio
    df['rEMA-15-5'] = df['EMA-15'] / df['EMA-5']

    # Mid-price
    df['mid_oc'] = 0.5 * (df['open'] + df['close'])
    df['mid_lh'] = 0.5 * (df['low']  + df['high'])

    # spreads
    df['delta_lh'] = df['high'] - df['low']
    df['delta_oc'] = df['close'] - df['open']
    df['delta_oc_abs'] = abs(df['open'] - df['close'])

    # Upper, Lower, Real body
    df['R'] = df['close'] - df['open']
    df['U'] = np.where(df['close'] > df['open'], df['high'] - df['close'], df['high'] - df['open'])
    df['L'] = np.where(df['close'] > df['open'],  df['open'] - df['low'], df['close'] - df['low'])

    # Candle direction (problem if used with normalization)
    df['updn'] = np.where(df['close'] > df['open'], 1, -1)


    ##
    df['RSI'] = ta.RSI(df)
    df['MFI'] = ta.MFI(df)
    df['ROC'] = ta.ROC(df)
    df['ADX'] = ta.ADX(df)
    df['ADOSC'] = ta.ADOSC(df)

    df['RSI-5']  = ta.RSI(df, timeperiod=5)
    df['RSI-15'] = ta.RSI(df, timeperiod=15)

    df['ADX-5']  = ta.ADX(df, timeperiod=5)
    df['ADX-15'] = ta.ADX(df, timeperiod=15)

    # Volume Flow Indicator (MFI) for volume based on the direction of price movement
    df['VFI'] = fta.VFI(df, period=14)

    # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
    rsi = 0.1 * (df['RSI'] - 50)
    df['FRSI'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

    # MACD
    macd = ta.MACD(df)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = macd['macd'], macd['macdsignal'], macd['macdhist']

    # Plus Directional Indicator / Movement
    df['PLUS_DM'] = ta.PLUS_DM(df)
    df['PLUS_DI'] = ta.PLUS_DI(df)

    # Minus Directional Indicator / Movement
    df['MINUS_DM'] = ta.MINUS_DM(df)
    df['MINUS_DI'] = ta.MINUS_DI(df)

    # Aroon, Aroon Oscillator
    aroon = ta.AROON(df)
    df['AROON_UP'], df['AROON_DN'] = aroon['aroonup'], aroon['aroondown']
    df['AROON_SC'] = ta.AROONOSC(df)

    # Stochastic Slow
    stoch = ta.STOCH(df)
    df['SLOWD'], df['SLOWK'] = stoch['slowd'], stoch['slowk']

    # Stochastic Fast
    stoch_fast = ta.STOCHF(df)
    df['FASTD'], df['FASTK'] = stoch_fast['fastd'], stoch_fast['fastk']

    # Parabolic SAR
    df['SAR'] = ta.SAR(df)

    # Bollinger Bands because obviously
    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=1)
    df['BB_low_20-1'], df['BB_mid_20-1'], df['BB_upp_20-1'] = bollinger['lower'], bollinger['mid'], bollinger['upper']
    df["BB_pct_20-1"] = ((df["close"] - df["BB_low_20-1"]) / (df["BB_upp_20-1"] - df["BB_low_20-1"]))
    df["BB_wid_20-1"] = ((df["BB_upp_20-1"] - df["BB_low_20-1"]) / df["BB_mid_20-1"])

    bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df['BB_low_20-2'], df['BB_mid_20-2'], df['BB_upp_20-2'] = bollinger['lower'], bollinger['mid'], bollinger['upper']
    df["BB_pct_20-2"] = ((df["close"] - df["BB_low_20-2"]) / (df["BB_upp_20-2"] - df["BB_low_20-2"]))
    df["BB_wid_20-2"] = ((df["BB_upp_20-2"] - df["BB_low_20-2"]) / df["BB_mid_20-2"])

    # # Bollinger Bands - Weighted (EMA based instead of SMA)
    # weighted_bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    # df["WBB_low"], df["WBB_mid"], df["WBB_upp"] = weighted_bollinger["lower"], weighted_bollinger["mid"], weighted_bollinger["upper"]
    # df["WBB_per"] = ((df["close"] - df["WBB_low"]) / (df["WBB_upp"] - df["WBB_low"]))
    # df["WBB_wid"] = ((df["WBB_upp"] - df["WBB_low"]) / df["WBB_mid"])

    # # Heikin Ashi Strategy
    # heikinashi = qtpylib.heikinashi(df)
    # df['ha_open'], df['ha_close'] = heikinashi['open'], heikinashi['close']
    # df['ha_high'], df['ha_low'] = heikinashi['high'],  heikinashi['low']

    # # Hilbert Transform Indicator - SineWave
    # hilbert = ta.HT_SINE(df)
    # df['htsine'] = hilbert['sine']
    # df['htleadsine'] = hilbert['leadsine']

    # Supertrend
    # add_supertrend(df, 7, 4)
    # add_supertrend(df, 10, 3)
    # add_supertrend(df, 14, 2)

    df['TR'] = ta.TRANGE(df)
    df['ATR-5'] = ta.SMA(df['TR'], 5)
    df['ATR-15'] = ta.SMA(df['TR'], 15)

    # smooth close price (don't use for predict because uses future information, only for labels!)
    smooth_window = 9
    df['sclose'] = savgol_filter(df['close'], smooth_window, 3, deriv=0)


    # auto-normalized features
    df['nf_close']  = (df['close'] / df['EMA-21'])
    df['nf_open']   = (df['open']  / df['EMA-21'])
    df['nf_low']    = (df['low']   / df['EMA-21'])
    df['nf_high']   = (df['high']  / df['EMA-21'])

    df['n_close']  = (df['close'] / df['EMA-55'])
    df['n_open']   = (df['open']  / df['EMA-55'])
    df['n_low']    = (df['low']   / df['EMA-55'])
    df['n_high']   = (df['high']  / df['EMA-55'])

    df['ns_close']  = (df['close'] / df['EMA-110'])
    df['ns_open']   = (df['open']  / df['EMA-110'])
    df['ns_low']    = (df['low']   / df['EMA-110'])
    df['ns_high']   = (df['high']  / df['EMA-110'])

    df['nRSI']  = df['RSI'] / 100
    df['nMFI']  = df['MFI'] / 100
    df['nROC']  = df['ROC'] / 100
    df['nADX']  = df['ADX'] / 100
    df['nMACD'] = df['MACD'] / 2000
    df['nVFI']  = df['VFI'] / 15

    df['nvolume'] = df['volume'] / (10 * ta.SMA(df['volume'], timeperiod=55))


    # week SMA
    c_week = ta.SMA(df['close'], timeperiod=672)

    df['no'] = df['open']  / c_week - 1
    df['nl'] = df['low']   / c_week - 1
    df['nh'] = df['high']  / c_week - 1
    df['nc'] = df['close'] / c_week - 1

    df['nR'] = df['R'] / c_week
    df['nL'] = df['L'] / c_week
    df['nU'] = df['U'] / c_week

    # slopes
    df['slope_w'] = df['close'].rolling(672).apply(slope, raw=False) / 100.
    df['slope_d'] = df['close'].rolling(96).apply(slope, raw=False)  / 100.
    df['slope_h'] = df['close'].rolling(4).apply(slope, raw=False)   / 100.

    # pct
    df['pct1_c'] = df['close'].pct_change(1)
    df['pct2_c'] = df['close'].pct_change(2)
    df['pct3_c'] = df['close'].pct_change(3)
    df['pct4_c'] = df['close'].pct_change(4)
    df['pct5_c'] = df['close'].pct_change(5)
    df['pct6_c'] = df['close'].pct_change(6)
    df['pct7_c'] = df['close'].pct_change(7)
    df['pct8_c'] = df['close'].pct_change(8)

    return df.copy()




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
    df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), -0.1,  0.1), 0)

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


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)



def compile_and_fit_c(name, model, w_train, w_valid, epochs=20, patience=5, bs=64, lr=0.0001, scheduler=None):

    cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=patience,
                                             mode='min',
                                             restore_best_weights=True)

    cb_cp = tf.keras.callbacks.ModelCheckpoint(filepath=f'model_{name}_best.h5',
                                           monitor='val_loss',
                                           mode='auto',
                                           save_best_only=True)

    callbacks = [cb_es, cb_cp,]
    if scheduler is not None:
        cb_lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
        callbacks.append(cb_lr)

    # def custom_acc(name, idx):
    #     m = tf.keras.metrics.BinaryAccuracy(name=name)
    #     def acc(y_true, y_pred):
    #         y_true_idx = tf.slice(y_true, begin=[0, idx], size=[-1,1])
    #         y_pred_idx = tf.slice(y_pred, begin=[0, idx], size=[-1,1])
    #         m.update_state(y_true_idx, y_pred_idx)
    #         return m.result()
    #     return m

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.optimizers.RMSprop(lr=lr),
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(name='acc'),
                  ])

    history = model.fit(w_train, epochs=epochs, validation_data=w_valid, batch_size=bs,
                        callbacks=callbacks, verbose=1)


    loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    ax[0].plot(epochs, loss, '.-', color='tab:blue', label='Training loss')
    ax[0].plot(epochs, val_loss, '.-', color='tab:red', label='Validation loss')
    ax[0].set_title('Training and validation loss')
    ax[0].legend()

    ax[1].plot(epochs, acc, '.-', color='tab:blue', label='Training acc')
    ax[1].plot(epochs, val_acc, '.-', color='tab:red', label='Validation acc')
    ax[1].set_title('Training and validation acc')
    ax[1].legend()

    return history


# class CustomAcc(keras.metrics.Metric):
#     def __init__(self, name="acc0", **kwargs):
#         super(CustomAcc, self).__init__(name=name, **kwargs)
#         self.true_positives_negatives = self.add_weight(name="ctp", initializer="zeros")
#         self.total                    = self.add_weight(name="ct", initializer="zeros")


#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))

#         y_true_idx = tf.slice(y_true, begin=[0, 0], size=[-1,1])
#         y_pred_idx = tf.slice(y_pred, begin=[0, 0], size=[-1,1])

#         values_true = y_true_idx>0
#         values_pred = y_pred_idx>0

#         self.true_positives_negatives.assign_add(tf.reduce_sum(tf.equal(values_true, values_pred)))
#         self.true_positives_negatives.assign_add(tf.reduce_sum(tf.equal(values_true, values_pred)))



#     def result(self):
#         return self.true_positives

#     def reset_states(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.true_positives.assign(0.0)

def compile_and_fit_r(name, model, w_train, w_valid, epochs=20, patience=5, bs=64, lr=0.0001,
                      scheduler=None):

    # rnn_model_2 = tf.keras.Sequential([
    #     tf.keras.layers.LSTM(32, return_sequences=False, input_shape=(past, n_features)),
    #     tf.keras.layers.Dense(future, activation='linear'),
    # ])

    cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=patience,
                                             mode='min',
                                             restore_best_weights=True)

    cb_cp = tf.keras.callbacks.ModelCheckpoint(filepath=f'model_{name}_best.h5',
                                           monitor='val_loss',
                                           mode='auto',
                                           save_best_only=True)

    callbacks = [cb_es, cb_cp,]
    if scheduler is not None:
        cb_lr = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
        callbacks.append(cb_lr)

    def custom_acc(name, idx):
        # m = tf.keras.metrics.BinaryAccuracy(name=name)
        def acc(y_true, y_pred):
            # y_true_idx = tf.cast(tf.where(tf.slice(y_true, begin=[0, idx], size=[-1,1]) > 0, 1, 0), tf.int32)
            # y_pred_idx = tf.cast(tf.where(tf.slice(y_pred, begin=[0, idx], size=[-1,1]) > 0, 1, 0), tf.int32)
            # m.update_state(tf.where(y_true_idx>0, 1, 0), tf.where(y_pred_idx>0, 1, 0))
            # return m.result()
            return tf.keras.backend.mean(tf.equal(tf.sign(y_true), tf.sign(y_pred)))
        return acc

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.RMSprop(lr=lr),
                  metrics=[tf.metrics.MeanAbsoluteError(name='mae'), custom_acc('acc', 0)])

    history = model.fit(w_train, epochs=epochs, validation_data=w_valid, batch_size=bs,
                        callbacks=callbacks)

    # plot training
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    mae = history.history['mae']
    val_mae = history.history['val_mae']

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1, 3, figsize=(18,6))

    ax[0].plot(epochs, loss, '.-', color='tab:blue', label='Training loss')
    ax[0].plot(epochs, val_loss, '.-', color='tab:red', label='Validation loss')
    ax[0].set_ylabel('Loss (MSE)')
    #ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].plot(epochs, mae, '.-', color='tab:blue', label='Training MAE')
    ax[1].plot(epochs, val_mae, '.-', color='tab:red', label='Validation MAE')
    ax[1].set_ylabel('MAE')
    #ax[1].set_yscale('log')
    ax[1].legend()

    ax[2].plot(epochs, acc, '.-', color='tab:blue', label='Training ACC')
    ax[2].plot(epochs, val_acc, '.-', color='tab:red', label='Validation ACC')
    ax[2].set_ylabel('ACC')
    #ax[2].set_yscale('log')
    ax[2].legend()

    return history


#
# Plots
#
def plot_prediction_r(window, model, npoints=100, random=False, save=None):

    fig, ax = plt.subplots(1, 1, figsize=(16,8))

    y_true, y_pred = window.get_true_and_pred_price(model, npoints, random)

    future = y_pred.shape[1]

    x = [ np.arange(i, npoints+i) for i in range(future+1) ]

    # true values
    ax.plot(x[0], y_true, marker='.', label='True', c='black')
    #ax.fill_between(x[0], y_l, y_h, label='Low/High', color='gray', alpha=0.5)

    # predicted values
    colors = list(mcolors.TABLEAU_COLORS)
    for i in range(future):
        ax.plot(x[i+1], y_pred[:,i], marker='.', color=colors[i], label=f'Prediction t+{i+1}')

    ax.legend(loc='upper left')
    ax.set_ylabel('Price', loc='top')
    ax.set_xlabel('Time', loc='right')

    if save is not None:
        fig.savefig(save)


def plot_prediction_r2(window, model, npoints=100, random=False, save=None):

    y_true, y_pred = window.get_true_and_pred_price(model, npoints, random)

    fig, ax = plt.subplots(1, 1, figsize=(16,8))

    future = y_pred.shape[1]
    x = np.arange(npoints)

    # true values
    ax.plot(x, y_true, marker='.', label='True', c='black', alpha=0.5)

    # predicted values
    for i in range(npoints):
        y = np.insert(y_pred[i,:], 0, y_true[i])
        ax.plot(np.arange(i, i+future+1), y, '--', lw=2, color='tab:orange')

    ax.legend(loc='upper left')
    ax.set_ylabel('Price', loc='top')
    ax.set_xlabel('Time', loc='right')

    if save is not None:
        fig.savefig(save)


def plot_prediction_r_diff(window, model, npoints=100, save=None):

    y_true, y_pred = window.get_true_and_pred(model, npoints, False)

    future = y_pred.shape[1]

    fig, ax = plt.subplots(future, 1, figsize=(16,16), sharex=True)

    if future == 1:
        ax = [ax,]

    width = 0.25

    for i in range(future):

        ax[i].bar(np.arange(npoints) - 0.25, y_true[:,i], width, align='edge', color='tab:blue', label=f'True t+{i+1}')
        ax[i].bar(np.arange(npoints) - 0.00, y_pred[:,i], width, align='edge', color='tab:red',  label=f'Pred t+{i+1}')

        ax[i].legend()

    ax[-1].set_ylabel('Diff', loc='top')
    ax[-1].set_xlabel('Time', loc='right')

    if save is not None:
        fig.savefig(save)


def plot_prediction_c(window, model=None, npoints=100, cut=0.5, random=False):

    y_true, y_pred = window.get_true_and_pred_price(model, npoints, random)

    future = y_pred.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x = np.arange(npoints)

    # true values
    ax.plot(x, y_true, '-', label='True', c='black', alpha=0.5)

    # predicted values
    y1_up, y1_dn = np.where(y_pred[:,0]>cut, y_true, np.nan), np.where(y_pred[:,0]<cut, y_true, np.nan)

    ax.scatter(x, y1_up, marker='o', color='tab:green')
    ax.scatter(x, y1_dn, marker='o', color='tab:red')

    for jj in range(1, future):
        y_up, y_dn = np.where(y_pred[:,jj]>cut, y_true, np.nan), np.where(y_pred[:,jj]<cut, y_true, np.nan)

        ax.scatter(x, (1+0.002*jj)*y_up, marker='o', color='tab:green')
        ax.scatter(x, (1+0.002*jj)*y_dn, marker='o', color='tab:red')

    ax.legend()
    ax.set_ylabel('Price', loc='top')
    ax.set_xlabel('Time', loc='right')


def plot_correlation(cols):
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(cols.corr(), ax=ax, cmap='RdBu', center=0)


def plot_cm(y_t, y_p):

    future = y_p.shape[1]

    fig, ax = plt.subplots(1, future, figsize=(5*future, 5))
    if future == 1:
        ax = [ax,]

    cmap = plt.get_cmap('Blues')

    for i in range(future):
        cm = confusion_matrix(y_t[:,i], np.where(y_p[:,i]>0.5, 1, 0))

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        ax[i].imshow(cm, interpolation='nearest', cmap=cmap)

        target_names = ['T', 'F']

        tick_marks = np.arange(len(target_names))
        ax[i].set_xticks(tick_marks)
        ax[i].set_xticklabels(target_names)
        ax[i].set_yticks(tick_marks)
        ax[i].set_yticklabels(target_names)

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        for ii, jj in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax[i].text(jj, ii, "{:0.4f}".format(cm[ii, jj]), horizontalalignment="center", color="black")

        ax[i].set_ylabel('True label')
        ax[i].set_xlabel('Predicted label') #\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))



def get_feature_importance(model, x, y_true, y_pred, j, n_features):

    from sklearn.metrics import accuracy_score
    s = accuracy_score(y_true, y_pred) # baseline score

    total = 0.0
    for i in range(n_features):

        perm = np.random.permutation(range(x.shape[0]))

        x_ = x.copy()
        x_[:, j] = x[perm, j]
        y_pred_ = clf.predict(X_test_)
        s_ij = accuracy_score(y_test, y_pred_)
        total += s_ij

    return s - total / n
