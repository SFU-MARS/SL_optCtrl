import pandas as pd
import os

import seaborn as sns
print(sns.__version__)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


class SL_valueLearner(object):
    def __init__(self):
        self.batch_size  = 50
        self.input_size  = 6 + 8
        # self.keep_prob   = 1
        # self.epoch = 50
        # self.train_data_size = 35000
        # self.test_data_size  = 15000
        self.train_data_size = 7000
        self.test_data_size  = 3000
    def read_data(self, use='train'):
        if use == 'train':
            data = pd.read_csv('./data/valueFunc_train.csv')
        elif use == 'test':
            data = pd.read_csv('./data/valueFunc_test.csv')
        length = len(data)

        X = []
        y = []

        for dx, values in data.iterrows():
            vect = []
            vect.extend(values[0:6])
            vect.extend(values[7:])
            # print("vect", vect)
            vect = np.array(vect)
            X.append(vect)

            v_gt = values[6]
            y.append([v_gt])

        X = np.array(X)
        y = np.array(y)

        print("read data for " + use + ' ...', np.shape(X))
        print("read label for " + use + ' ...', np.shape(y))

        return X,y


    # def dense_layer(input, in_size, out_size, activation_function=None):
    #     Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #     biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    #     Wx_plus_b = tf.matmul(input, Weights) + biases  # not actived yet
    #     Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=keep_prob_s)
    #     if activation_function is None:
    #         output = Wx_plus_b
    #     else:
    #         output = activation_function(Wx_plus_b)
    #     return output
    #
    #
    # def build_graph(self):
    #     self.xs = tf.placeholder(tf.float32, [None, self.input_size])
    #     self.ys = tf.placeholder(tf.float32, [None, 1])
    #
    #     # TODO: maybe need to normalize input xs
    #     # xs = xs.normalize()
    #     self.hidden_out1 = dense_layer(xs, self.input_size, 64, activation_function=tf.nn.tanh)
    #     self.hidden_out2 = dense_layer(hidden_out1, 64, 64, activation_function=tf.nn.tanh)
    #     self.vpred = dense_layer(hidden_out2, 64, 1, activation_function=None)
    #
    #     self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - vpred), reduction_indices=[1]))
    #     self.train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
    #
    # def train(self):
    #     X_train, y_train = read_data(use='train')
    #
    #     with tf.Session() as sess:
    #         init = tf.initialize_all_variables()
    #         sess.run(init)
    #
    #         # total iter_num = data_size / batch_size * epoch_num
    #         for iter in range(self.epoch * self.train_data_size / self.batch_size):
    #             start = (iter * batch_size) % self.train_data_size
    #             end   = min(start + batch_size, self.train_data_size)
    #
    #             _, pred, loss = sess.run([train_step, vpred, loss], feed_dict={xs: X_train[start:end], ys: y_train[start:end], keep_prob_s: keep_prob})
    #
    #             if iter % 50 == 0:
    #                 print("iter: ", '%04d' % (iter + 1), "loss: ", los)
    #
    # def test(self):
    #     X_test, y_test = read_data(use='test')
    #
    #     # TODO: load trained model for test

def build_model(train_dataset):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.tanh, input_shape=[len(train_dataset.keys())]),
        keras.layers.Dense(64, activation=tf.nn.tanh),
        keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer = optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


if __name__ == "__main__":

    dirpath = os.path.dirname(__file__)
    dataset_path = None
    if not os.path.exists(dirpath + "/auto-mpg.data"):
        dataset_path = keras.utils.get_file(dirpath + "/auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
        print("dataset_path:", dataset_path)
    else:
        dataset_path = dirpath + "/auto-mpg.data"
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight', 'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    print(dataset.tail())

    print(dataset.isna().sum())

    dataset = dataset.dropna()
    origin  = dataset.pop('Origin')

    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0
    print(dataset.tail())

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset  = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()


    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')





    model = build_model(train_dataset)
    model.summary()
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data  = norm(test_dataset, train_stats)


    # sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    #
    # plt.show()



    EPOCHS = 1000

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    plot_history(history)

