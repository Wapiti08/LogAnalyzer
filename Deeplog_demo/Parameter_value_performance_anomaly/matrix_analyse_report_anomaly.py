# !/usr/bin/env python
# coding: utf-8


# baseline approach
# according to the paper: DeepLog: Anomaly Detection and Diagnosis from System Logs
#                                 through Deep Learning
# the baseline approach is not available
# Deeplog approach ----input: normalize (mean), standard deviation
# output: a real value vector as a prediction for the next parameter
# training stage: mean square loss ---- minimize the error
# Anomaly detection: mse loss metrics between prediction ----- gaussian distribution expresses the errors
# how to detect: the error is within a high-level of confidence interval ---- normal. otherwise abnormal

# a separate LSTM network is built for the parameter
# value vector sequence of each distinct log key value


'''
This part is to analyse the log key matrix and report the anomaly logs
'''

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, MinMaxScaler
import re
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.metrics import mean_squared_error
from pandas import Series
from math import sqrt, pow
from numpy import concatenate, subtract
from pandas import DataFrame
from pandas import concat
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn import model_selection
import os
import joblib
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest
from visualized_value_vector import visualize_value


# ======================= generate the normal rmses.pkl and rmses_dict.pkl ================================

def mean_squared_error_modified(y_true, y_pred):
    d_matrix = subtract(y_true, y_pred)
    print("the d_matrix is:", d_matrix)
    means = []
    for i in range(d_matrix.shape[1]):
        means.append(np.mean(d_matrix[:, i] * d_matrix[:, i], axis=-1))
    print("the means are:", means)
    return np.mean(means), means


def training_data_generate(matrix, n_steps):
    '''
    :param matrix: the paramter value vectors for a single log key
    :param n_steps_in: the length of sequence, which depends on how long the matrix is
    :param n_steps_out: always one, we just need one really parameter vector
    :return:
    '''
    X, Y = list(), list()
    for i in range(matrix.shape[0]):
        # find the end of this pattern
        end_ix = i+n_steps
        # check whether beyond the dataset
        if end_ix > matrix.shape[0]-1:
            break
        seq_x, seq_y = matrix[i:end_ix,:], matrix[end_ix,:]
        X.append(seq_x)
        Y.append(seq_y)
    X, Y = np.array(X), np.array(Y)
    print("the shape of X is:",X.shape)
    return X, Y


def LSTM_model(trainx, trainy):
    # use the train
    model = Sequential()
    model.add(LSTM(100, activation = 'relu', return_sequences = True, input_shape=(trainx.shape[1], trainx.shape[2])))
    print("the train x is {} and its shape is {}".format(trainx, trainx.shape))
    model.add(LSTM(100, activation = 'relu'))
    model.add(Dense(trainx.shape[2]))
    model.compile(loss='mse', optimizer='adam')
    # model.fit(trainx, trainy, epochs = 50, verbose=2, callbacks=[callbacks])
    model.fit(trainx, trainy, epochs=50, verbose=2)
    model.summary()
    joblib.dump(model,'model.pkl')
    return model


if __name__ == "__main__":
    # use loop to input the name of matrix and get corresponding dataframe
    filenames = []
    root_dir = '../Dataset/Linux/Malicious_Separate_Structured_Logs/Event_npy/'
    # r=root, d = directories, f=files
    for r, d, f in os.walk(root_dir):
        for file in f:
            if file.endswith('.npy'):
                filenames.append(os.path.join(r, file))
    # set the random seed
    seed = 7
    rmses = []
    rmses_dict = {}
    # define the normal and abnormal log lists
    normal_key_log = []
    abnormal_key_log = []
    # identify whether result file has been generated before
    for file in filenames:
        if os.path.isfile(file+'_rmses.pkl'):
            rmses = joblib.load(file+'_rmses.pkl')

        else:
            # looping read single file
            print("we are processing matrix:", file)
            matrix = np.load(file)
            # set n_steps_in and n_steps_out depending on the sequence length of matrix
            # we set the test_size=0.4, the length of matrix should be at least 8
            # Here, I will change the length of history to see the performance
            if matrix.shape[0] >= 8:
                n_steps = 3
                X, Y = training_data_generate(matrix, n_steps)
                train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.4, random_state=seed)

            elif matrix.shape[0]>=4:
                n_steps = 1
                X, Y = training_data_generate(matrix, n_steps)
                # test_x and
                train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.5, random_state=seed)
            else:
                continue
            # get the model
            model = LSTM_model(train_x, train_y)
            print("the test_x is:", test_x)
            # make a prediction
            yhat = model.predict(test_x)
            # delete the time step element
            print("the predicted y is:", yhat)

            rmse, means = mean_squared_error_modified(test_y, yhat)
            # rmse, meams = mean_squared_error(test_y, yhat)
            rmse = sqrt(rmse)
            print('Test RMSE: %.3f' % rmse)
            # use the mean square error to compare the difference between predicted y and validation y
            # the error follows the Gaussian distribution ---- normal, otherwise abnormal
            rmses.append(rmse)
            # save the result
            rmses_dict[file] = rmse
            # save the results to files
            joblib.dump(rmses, file+'_rmses.pkl')
            joblib.dump(rmses_dict, file+'_rmses_dict.pkl')

        # use shapiro to calculate the similarity between the errors and a Gaussian distribution
        # suitable for examples above 3


        # here, the problem is there may be not only one gaussian distribution, so we should find the way
        # to improve the matching method

        # if len(rmses) >= 3:
        #     stat, p = shapiro(rmses)
        #     print("Statistics = %.3f, p = %.3f"%(stat, p))
        #     # the threshold for similarity
        #     alpha = 0.05
        #
        #     if p > alpha:
        #         print("Sample looks Gaussian and {} is like the normal log".format(file))
        #         normal_key_log.append(file)
        #     else:
        #         print("Sample does not look Gaussian and {} might be abnormal log".format(file))
        #         abnormal_key_log.append(file)
        # else:
        #     continue

        # part to check whether the coming log entry will follow the confidence interval of Gaussian Distribution



        # part to print the picture of means with bar chart
        # create the x axis labels for plot
        x_list = []
        for i in range(len(rmses)):
            x_list.append(i)
        # plt.bar(x_list, rmses)
        # plt.ylabel("Errors Values")
        # plt.title(file+' '+'Errors Distribution')
        # plt.show()

        # part to print the picture of means with line chart
        plt.plot(x_list, rmses)
        plt.ylabel("Errors Values")
        file_number = re.findall('\d+',file)
        print("the file_number is:",file_number)
        plt.title(file_number[0] + ' ' + 'Errors Distribution')
        # plt.title(file + ' ' + 'Errors Distribution')
        plt.show()


        #  use normaltest to calculate the similarity between the errors and a Gaussian distribution
        #  suitable for examples above 8
        # if len(rmses) >= 3:
        #     stat, p = normaltest(rmses)
        #     print("Statistics = %.3f, p = %.3f"%(stat, p))
        #     # the threshold for similarity
        #     alpha = 0.05
        #
        #     if p > alpha:
        #         print("Sample looks Gaussian and {} is like the normal log".format(file))
        #         normal_key_log.append(file)
        #     else:
        #         print("Sample does not look Gaussian and {} might be abnormal log".format(file))
        #         abnormal_key_log.append(file)
        # else:
        #     continue
        print("the rmses_dict is {}".format(rmses_dict))
        print("the mean of rmses is: {}".format(np.mean(rmses)))

    print("the normal key log list is:", normal_key_log)
    print("the abnormal key log list is:", abnormal_key_log)



    '''
    with history--3, random_state--7, threshold_alpha--0.05, simulation -- shapiro
        we got the result: the normal key log list is: ['Event_npy/E0.npy', 'Event_npy/E12.npy', 'Event_npy/E117.npy']
    
    From experiments we know:
    1. random_state doesn't make an influence on the final result (maybe the dataset is not large enough )
    2. The history length has no influence on the result
    3. The value of alpha has no influence on the result.
    
    '''