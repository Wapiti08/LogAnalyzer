# the principles behind Execution path anomaly
# 1. first, we generate the set of distinct log keys from our program
# 2. parse the entries into log keys, the log key sequence ---- an execution path
# 3. the model DeepLog is a multi-class classifier over recent context
## 1. input the recent log keys
## 2. a probability distribution over the n log keys from K ----
## the probability that the next log key in the sequence is a key ki belongs to K

'''

LSTM
it learns a probability distribution Pr(mt=ki | M(t-h),...M(t-1)) that maximizes the
prob of the training log key sequence

every single block remembers a state for its input as a vector of a fixed dimension

the output of previous block+data input -----> fed into this one block

one layer(h unrolled LSTM blocks) ---> a series of LSTM blocks ---- each
cell includes a
hidden vector H(t-i) and cell state vector C(t-i）

'''

'''
Description about the model:
each block remembers a state for its input as a vector of a fixed dimension

input: a window w of h log keys --- (w = {mt−h, . . . ,mt−1})
output: the log key value comes right after w

the loss function will be categorical cross-entropy loss

omits the input layer and and output layer ---- encoding-decoding schemes
the input layer encodes the n possible log keys from K(log keys set) as one-hot vectors

output layer:  a standard multinomial logistic function to represent Pr[mt = ki|w]
'''

import pandas as pd
from keras.models import Sequential
import keras
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from nltk.probability import FreqDist
import nltk
from keras.utils import *
import numpy as np
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
import os
from sklearn.metrics import mean_squared_error

# =================== build the LSTM part for the first model DeepLog ============================

def load_value_vector(filename):
    # get the normal system execution path
    df = pd.read_csv(log_value_vector_path)
    df = df.copy()
    return df

# function to transfer log key into EventId
def key_to_EventId(df):
    log_key_sequence = df['log key']
    log_key_sequence = list(log_key_sequence)
    # get the unique list
    items = set(log_key_sequence)
    # define the total number of log keys
    K = None
    K = len(items)
    print("the length of log_key_sequence is:", len(items))
    key_name_dict = {}

    for i, item in enumerate(items):
        # items is a set
        # columns are the lines of log key sequence
        for j in range(len(log_key_sequence)):
            if log_key_sequence[j] == item:
                name = 'E' + str(i)
                # log_key_sequence[j]='k'+str(i)
                key_name_dict[name] = log_key_sequence[j].strip('\n')

    joblib.dump(key_name_dict,'Deeplog/path_key_name_dict.pkl')

    return log_key_sequence, key_name_dict, K

# function to replace the log key to eventID in a log key sequence
def transform_key_k(log_key_sequence, dict):
    while set(log_key_sequence) == set(dict.values()):
        for key, value in key_name_dict.items():
            for x in log_key_sequence:

                if value == x:

                    log_key_sequence[log_key_sequence.index(x)] = str(key)
                else:
                    continue
        return log_key_sequence


# function to filter E in a str and get the sequence with 3:1
def get_train(log_key_sequence_str):
    #     # we have the sequence of log keys
    #     seq = np.array(log_key_sequence)
    # divide the log sequence into 4 for every unit
    tokens = log_key_sequence_str.split(' ')
    for i in range(len(tokens)):
        tokens[i] = tokens[i].replace('E', '')
        tokens[i] = int(tokens[i])
    #     print("the tokens are:",tokens)
    bigramfdist_4 = FreqDist()
    bigrams_4 = ngrams(tokens, 4)

    bigramfdist_4.update(bigrams_4)
    # print("the bigramfdsit_4 is:",bigramfdist_4.keys())
    # we set the length of history logs as 3
    seq = np.array(list(bigramfdist_4.keys()))
    # print("the seq is:",seq)
    X, Y = seq[:, :3], seq[:, 3:4]

    return X, Y


# ================= part to generate the training data ======================
# function of callback
class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.8):
            print("Reached 80% accuracy so stopping training")
            self.model.stop_learning = True

# # ============  Implement the lstm model ==================

def lstm_model(x, y ,callbacks):
    batch_size = 16
    # according to the article, we will stack two layers of LSTM, the model about stacked LSTM for sequence classification
    model = Sequential()
    # =============== model 1 ===================
    # input data shape: (batch_size, timesteps, data_dim)
    model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    # output layer with a single value prediction (1,K)
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    # to ensure the training data patterns remain sequential --- disable the shuffle
    # make it stateful, we add batch_size
    model.fit(x, y, epochs=500, batch_size=batch_size, verbose=2, callbacks=[callbacks], shuffle=False)
    # to see the summary of input and output shape
    model.summary()
    print('the accuracy for single lstm model is:', model.evaluate(x, y, batch_size=batch_size, verbose=0))
    joblib.dump(model, 'Deeplog/path_anomaly_model.pkl')
    return model

# ============ part to prediction ==============
def model_predict(model, x_test, y_test):
    errors = None
    # predict the x_test
    Y_pred = model.predict(x_test, verbose=0)
    errors = mean_squared_error(Y_pred, y_test)
    print("the errors are:",errors)
    # for i in range(Y_pred.shape[1]):
        # print("the index of predicted one_hot_labels {} are: {}".format(Y_pred[i],np.argmax(Y_pred[i])))

def model_predcit_trace(model, x_test, y_test, key_name_dict):
    anomaly_log_keys = []
    y_pred = model.predict(x_test, verbose = 0)
    yhat = []
    for i in range(y_pred.shape[0]):
        yhat.append(np.argmax(y_pred[i]))

    print('the length of yhat {} is {}'.format(yhat, len(yhat)))

    for n in range(len(yhat)):
        if yhat[n] == np.argmax(y_test[n]):
            pass
        else:
            eventId = 'E'+ str(n)
            anomaly_log_key = key_name_dict[eventId]
            anomaly_log_keys.append(anomaly_log_key)
            print("log {} is possible anomaly".format(anomaly_log_key))
    joblib.dump(anomaly_log_keys, 'Deeplog/anomaly_log_key.pkl')


if __name__ == "__main__":
    # define the csv path needed to be loaded
    log_value_vector_path = '../data/System_logs/log_value_vector.csv'
    df = load_value_vector(log_value_vector_path)

    # check whether the key_name_dict has been generated
    key_name_dict_path = 'Deeplog/path_key_name_dict.pkl'

    if os.path.isfile(key_name_dict_path):
        print("key_name_dict file has been generated before")
        key_name_dict = joblib.load(key_name_dict_path)
        log_key_sequence = df['log key']
        # get the unique log_key_sequence set
        log_key_sequence = list(log_key_sequence)
        items = set(log_key_sequence)
        # define the number of clusters
        K = len(items)
    else:
        log_key_sequence, key_name_dict, K = key_to_EventId(df)

    # get the EventID sequence
    log_key_id_sequence = transform_key_k(log_key_sequence, key_name_dict)

    # transfrom the list of data to str data
    for i in range(len(log_key_sequence)):
        log_key_sequence_str = ' '.join(log_key_sequence)

    # get the raw training data
    X_normal, Y_normal = get_train(log_key_sequence_str)

    # reshape the X_normal to make it suitable for training
    X_normal = np.reshape(X_normal, (-1, 3, 1))
    # normalize
    X_normal = X_normal / K
    Y_normal = keras.utils.to_categorical(Y_normal, num_classes=K)
    x_train, x_test, y_train, y_test = train_test_split(X_normal, Y_normal, test_size=0.3, random_state=0)
    print("the lengths of training data and testing data is {} and {}".format(x_train.shape[0], x_test.shape[0]))
    # check whether the model has existed
    filename = 'Deeplog/path_anomaly_model.pkl'
    # load the callback example
    callbacks = Mycallback()
    if os.path.isfile(filename):
        print("model has been generated before")
        model = joblib.load(filename)
        model_predict(model, x_test,y_test)
        model_predcit_trace(model, x_test, y_test, key_name_dict)
    else:
        model = lstm_model(x_train, y_train, callbacks)
        model_predict(model, x_test,y_test)
        model_predcit_trace(model, x_test, y_test, key_name_dict)


'''
the accuracy for single lstm model is: [0.9844519591639402, 0.6494464949048313]
'''

