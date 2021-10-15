import math
import numpy as np
import tensorflow as tf


class WindowGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, input_width, label_width,
                 features,
                 batch_size=64,
                 reverse=False, shuffle=False,
                 norm_features=['close'],
                 labels=[], label_type='int'):

        self.data = df ##.copy() ##[features+label]

        self.batch_size = batch_size
        self.reverse = reverse

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.total_window_size = input_width + label_width

        # Input features
        self.features = features
        self.column_idxs = { name: i for i, name in enumerate(features) }

        self.norm_features = norm_features
        self.norm_features_idxs = [ self.column_idxs[name] for name in norm_features ]

        # Label
        self.labels = labels
        self.label_type = label_type
        # self.label_cols = [ f'label_t+{i}' for i in range(1, self.label_width+1) ]

        #
        steps = ((len(self.data)-self.total_window_size) // batch_size)

        # remove windows to match bs * steps
        n = batch_size * steps + self.total_window_size
        self.data.drop(self.data.tail(len(self.data)-n-1).index, inplace=True)

        # total posible windows
        self.total_windows = batch_size * steps
        self.steps = steps

        # self.label_col = label_col
        # self.add_labels(label_bias, label_col)

        print(f'WindowGenerator: total={len(self.data)}, total_windows={self.total_windows}, batch_size={batch_size}, batches={steps}, labels={self.labels}, shuffle={shuffle}, n_features={len(self.features)}')

        self.idxs = np.arange(self.total_windows)

        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.idxs)


    # def add_labels(self, col):

    #     if self.label == 'avgN':
    #         delta = (self.data[col].shift(-i) - self.data[col]) / self.data[col]


    #     for i in range(1, self.label_width+1):
    #         delta = (self.data[col].shift(-i) - self.data[col]) / self.data[col]

    #         if self.label == 'cdiff':
    #             self.data[f'label_t+{i}'] = np.where(delta>bias, 1, 0)
    #         elif self.label == 'ndiff':
    #             self.data[f'label_t+{i}'] = delta
    #         elif self.label == 'avgN':
    #             self.data[f'label_t+{i}'] = delta

    def __len__(self):
        return self.steps

    def _get_data(self, i):

        idx_beg = self.idxs[i]
        idx_end = idx_beg + self.input_width - 1

        # input
        inputs = self.data.iloc[idx_beg:idx_end+1][self.features].to_numpy()

        for idx in self.norm_features_idxs:
            w_mean, w_std = inputs[:,idx].mean(), inputs[:,idx].std()
            inputs[:,idx] = (inputs[:,idx] - w_mean) / (10*w_std)

        # for idx in self.norm_features_idxs:
        #     wmin, wmax = inputs[:,idx].min(), inputs[:,idx].max()
        #     inputs[:,idx] = (inputs[:,idx] - wmin) / (wmax-wmin)

        # label
        labels = self.data.iloc[idx_end][self.labels].to_numpy()

        labels = labels.reshape((len(self.labels), 1))

        if self.reverse:
            inputs = np.flip(inputs, axis=0)
            labels = np.flip(labels, axis=0)

        return inputs, labels

    def __getitem__(self, index):

        batch_x = np.zeros((self.batch_size, self.input_width, len(self.features)), 'float')
        batch_y = np.zeros((self.batch_size, self.label_width, 1), self.label_type)

        beg = index * self.batch_size
        for i in range(self.batch_size):
            batch_x[i], batch_y[i] = self._get_data(beg+i)

        if self.label_type == 'int' and len(self.labels) == 1:
            batch_y = batch_y.squeeze()

        return batch_x, batch_y


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)


    #
    def get_prediction(self, model, i):
        """
        get predictions from i for i+1, i+2, ..., i+n
        """

        inputs = self.data.iloc[i-self.input_width+1:i+1][self.features].to_numpy()

        # rescale input
        for idx in self.norm_features_idxs:
            w_mean, w_std = inputs[:,idx].mean(), inputs[:,idx].std()
            inputs[:,idx] = (inputs[:,idx] - w_mean) / (10*w_std)

        # predict
        yp = model.predict(np.expand_dims(inputs, axis=0)).squeeze()

        return yp


    def get_true_and_pred(self, model, npoints, random):

        start = len(self.data) - npoints

        y_true = self.data.iloc[start:][self.label_cols].to_numpy()

        y_pred = np.empty((npoints, self.label_width))
        for i in range(npoints):
            y_pred[i,:] = self.get_prediction(model, start+i)

        return y_true, y_pred


    def get_true_and_pred_price(self, model, npoints, random):

        start = len(self.data) - npoints

        y_true = self.data.iloc[start:][self.label_col].to_numpy()

        y_pred = np.empty((npoints, self.label_width))
        for i in range(npoints):
            if self.label == 'ndiff':
                delta_p = self.get_prediction(model, start+i)
                y_pred[i,:] = (y_true[i] * (delta_p)) + y_true[i]
            else:
                y_pred[i,:] = self.get_prediction(model, start+i)

        return y_true, y_pred


def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

def cos_sum(a, b):
    return math.cos(a+b)

class WindowGenerator2D(WindowGenerator):

    def __init__(self, df, input_width, label_width,
                 features, batch_size=64,
                 shuffle=False, labels=[]):

        super().__init__(df, input_width, label_width, features,
                         batch_size=batch_size,
                         shuffle=shuffle, labels=labels)

        self.nfeatures = len(features)

    def _get_data(self, i):

        idx_beg = self.idxs[i]
        idx_end = idx_beg + self.input_width - 1

        # input
        inputs = self.data.iloc[idx_beg:idx_end+1][self.features].to_numpy()

        #input_imgs = np.empty((self.nfeatures, self.input_width, self.input_width))
        #for idx in range(self.nfeatures):

        img1 = self.get_GAF(inputs[:,0])
        img2 = self.get_GAF(inputs[:,1])
        img3 = self.get_GAF(inputs[:,2])
        img4 = self.get_GAF(inputs[:,3])

        # label
        labels = self.data.iloc[idx_end][self.labels].to_numpy()
        labels = labels.reshape((len(self.labels), 1))

        return img1, img2, img3, img4, labels

    def __getitem__(self, index):

        batch_x1 = np.empty((self.batch_size, self.input_width, self.input_width), 'float')
        batch_x2 = np.empty((self.batch_size, self.input_width, self.input_width), 'float')
        batch_x3 = np.empty((self.batch_size, self.input_width, self.input_width), 'float')
        batch_x4 = np.empty((self.batch_size, self.input_width, self.input_width), 'float')

        batch_y = np.empty((self.batch_size, len(self.labels), 1), 'float')

        beg = index * self.batch_size
        for i in range(self.batch_size):
            batch_x1[i], batch_x2[i], batch_x3[i], batch_x4[i], batch_y[i] = self._get_data(beg+i)

        return (batch_x1, batch_x2, batch_x3, batch_x4), batch_y

    # GAF
    def get_GAF(self, series):
        """Compute the Gramian Angular Field of an image"""

        # Min-Max scaling
        min_ = np.amin(series)
        max_ = np.amax(series)
        scaled_series = (2*series - max_ - min_)/(max_ - min_)

        # Floating point inaccuracy!
        scaled_series = np.where(scaled_series >= 1., 1., scaled_series)
        scaled_series = np.where(scaled_series <= -1., -1., scaled_series)

        # Polar encoding
        phi = np.arccos(scaled_series)
        # Note! The computation of r is not necessary
        # r = np.linspace(0, 1, len(scaled_series))

        # GAF Computation (every term of the matrix)
        gaf = tabulate(phi, phi, cos_sum)

        return gaf ##, phi, r, scaled_series)
