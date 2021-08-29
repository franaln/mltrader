import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import utils

class WindowGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, input_width, label_width,
                 features, label_col='close',
                 batch_size=64):

        self.batch_size = batch_size

        self.features = features
        self.column_idxs = { name: i for i, name in enumerate(features) }

        self.label_idx = self.column_idxs[label_col]

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width

        self.total_window_size = input_width + label_width

        self.input_slice = slice(0, input_width)
        self.input_idxs  = np.arange(self.total_window_size)[self.input_slice]

        self.label_start  = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_idxs   = np.arange(self.total_window_size)[self.labels_slice]

        self.skip_normalization = [ i for i, name in enumerate(features) if name in utils.indicators_wo_normalization ]

        #
        data = np.array(df[features], dtype=np.float32)

        steps = ((len(data)-self.total_window_size) // batch_size)

        # remove windows to match bs * steps
        n = batch_size * steps + self.total_window_size
        self.data = data[:n,:]

        # total posible windows
        self.total_windows = batch_size * steps
        self.steps = steps


    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_idxs}',
            f'Label indices: {self.label_idxs}'])


    def get_mean_std(self, inputs):

        w_mean = inputs.mean(axis=0)
        w_std  = inputs.std(axis=0)

        for i in self.skip_normalization:
            w_mean[i] = 0
            w_std[i] = 1

        return w_mean, w_std

    def scale(self, inputs, labels):

        w_mean, w_std = self.get_mean_std(inputs)

        inputs = (inputs - w_mean) / w_std
        labels = (labels - w_mean[self.label_idx]) / w_std[self.label_idx]

        return inputs, labels

    def __len__(self):
        return self.steps

    def _split_window(self, window):
        inputs = window[self.input_slice, :]
        labels = window[self.labels_slice, self.label_idx]
        return self.scale(inputs, labels)

    def __get_data(self, i):

        window = self.data[i:i+self.total_window_size]

        inputs, labels = self._split_window(window)

        return inputs, labels.reshape((self.label_width, 1))

    def __getitem__(self, index):

        batch_x = np.empty((self.batch_size, self.input_width, len(self.features)), 'float')
        batch_y = np.empty((self.batch_size, self.label_width, 1), 'float')

        start = index * self.batch_size

        for i in range(self.batch_size):
            batch_x[i], batch_y[i] = self.__get_data(start+i)

        return batch_x, batch_y


    def plot(self, ax, model=None, col='close', n=100):

        start = len(self.data) - n

        x = [ np.arange(i, n+i) for i in range(self.label_width) ]

        y_real = np.array(self.data[start:,self.column_idxs[col]])

        y_pred = np.empty((n, self.label_width))
        for i in range(n):

            inputs = self.data[start+i-self.input_width:start+i]

            w_mean, w_std = self.get_mean_std(inputs)

            # rescale input and predict
            inputs = (inputs - w_mean) / w_std
            yp = model.predict(np.expand_dims(inputs, axis=0))

            # rescale output
            yp = (yp * w_std[self.label_idx]) + w_mean[self.label_idx]
            y_pred[i,:] = yp[0,:]


        # true values
        ax.plot(x[0], y_real, marker='.', label='True values', c='black')

        # predicted values
        colors = list(mcolors.TABLEAU_COLORS)

        for i in range(self.label_width):
            ax.plot(x[i], y_pred[:,i], marker='.', color=colors[i], label=f'Prediction t+{i+1}')


        ax.legend(loc='upper left')

        ax.set_ylabel('Close price', loc='top')
        ax.set_xlabel('Time [15m]', loc='right')
