# !/usr/bin/env python
# coding: utf-8


# baseline approach
# according to the paper: DeepLog: Anomaly Detection and Diagnosis from System Logs
#                                 through Deep Learning
# the baseline approach is not available
# Deeplog approach ----input: normalize (mean), standard deviation
# output: a real value vector as a prediction for the next parameter
# training stage: mean square loss ---- minimize the error
# anomaly detection: mse loss metrics between prediction ----- gaussian distribution expresses the errors
# how to detect: the error is within a high-level of confidence interval ---- normal. otherwise abnormal

# a separate LSTM network is built for the parameter
# value vector sequence of each distinct log key value

'''
executing instruction:
In total, there are two parts.
For every part, there are three steps: when you execute one step in a part, please comment the left two steps
please execute a part every time
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

# ======================= generate the normal rmses.pkl and rmses_dict.pkl ================================
# ==================== Step 1 =====================
# ===================== part to visualize to normal logs ======================
fd = pd.read_csv('../data/System_logs/log_value_vector.csv')


# parameter_value_vectors = []
# # get the parameter_value_vector line
# parameter_value_vectors = fd['parameter value vector']
# time_gap_lists = []
# # copy the orginal data used for analysis
# time_gap_lists = parameter_value_vectors.copy()
# time_gap_lists = [var.split(',')[0] for var in time_gap_lists]
# # transfer the str data into int dtype
# replace_pattern = { "[": "", "]": ""}
# # define the function to replace multiple values
# def replace_all(text, dic):
#     for i, j in dic.items():
#         text = text.replace(i, j)
#     return text
# # replace the '[' and ']' in a string
# time_gap_lists = [int(replace_all(var, replace_pattern)) for var in time_gap_lists]
# time_gap_lists
#
# plt.hist(time_gap_lists,bins=100)
# plt.xlabel('Time Gap')
# plt.ylabel('Occurrence')
# plt.title('Normal Linux Time Anomaly Detection')


# build lstm model
# Input: need the normalization and standard deviation
# Output: real value vector as a prediction for next parameter value vector
# optimize the LSTM model ---- mean square loss（MSE）
# Anomaly detection: training set and the validation set
# Judging criteria: if the error is within interval of Gaussian Distribution ---- normal, otherwise abnormal


# def key_to_EventId(df):
#     '''
#     :param df: normaly, the log key column in df is hashed values
#     :return: log_key_sequence: the column of log key
#              key_name_dict: format is {Exx: SRWEDFFW(hashed value),...}
#              K: the number of unique log key events
#     '''
#     log_key_sequence = df['log key']
#     log_key_sequence = list(log_key_sequence)
#     # get the unique list
#     items = set(log_key_sequence)
#     # define the total number of log keys
#     K = None
#     K = len(items)
#     print("the length of log_key_sequence is:", len(items))
#     key_name_dict = {}
#
#     for i, item in enumerate(items):
#         # items is a set
#         # columns are the lines of log key sequence
#         for j in range(len(log_key_sequence)):
#             if log_key_sequence[j] == item:
#                 name = 'E' + str(i)
#                 # log_key_sequence[j]='k'+str(i)
#                 key_name_dict[name] = log_key_sequence[j].strip('\n')
#
#     return log_key_sequence, key_name_dict, K
# #
# #
# # # ================= get the vocabulary set ==================
# #
# def vocalubary_generate(fd):
#     '''
#     :param fd:  pandas dataframe with the log key column in it is hashed values
#     :return: fd_id: copied fd dataframe, in order to protect the original data
#              key_para_dict: the format is {Exx:[textual parameter 1],[textual parameter 2],...}
#     '''
#
#     key_para_dict = {}
#
#     log_key_sequence, key_name_dict, K = key_to_EventId(fd)
#     fd_id = fd.copy()
#     # swith the key and value in a dict
#     key_name_dict_rev = dict((value,key) for key,value in key_name_dict.items())
#     # mapping the value to keyID
#     fd_id['log key'] = fd_id['log key'].map(key_name_dict_rev)
#
#     uni_log_key_id = list(set(fd_id['log key']))
#
#     parameters = []
#
#     for i in range(len(uni_log_key_id)):
#         # get all the parameters with the same eventID
#         parameters = fd_id[fd_id['log key'] == uni_log_key_id[i]]['parameter value vector']
#         key_para_dict[uni_log_key_id[i]] = parameters.values[:]
#     # print("the length of array is:",[len(var) for var in key_para_dict.values()])
#     # padding nan to object without enough length
#     df_dict_para = pd.DataFrame(dict([(k,Series(v)) for k,v in key_para_dict.items()]))
#     df_dict_para.to_csv('../data/System_logs/key_para_dict.csv',index= False, header=key_para_dict.keys())
#     return key_para_dict, fd_id
#
# # input the normal fd file
# key_para_dict, fd_id = vocalubary_generate(fd)
# # input the abnormal fd file
# # key_para_dict, fd_id = vocalubary_generate(fd_mali)
#
#
# def tokens_generate(key_para_dict):
#     '''
#     :param key_para_dict: the format is {Exx:[textual parameter 1],[texual parameter 2],...}
#     :return: tokens: all the word tokens in the parameter value vector column
#     '''
#     text = []
#     for key, value in key_para_dict.items():
#         # extract the time part from values
#         for i in range(len(value[:])):
#             if value[i].split(',')[1:] == []:
#                 break
#             else:
#                 value[i] = re.sub('[\[|\]|\'|\|\s+|\.|\-]', '', str(value[i])).split(',')
#                 if value[i] == ['']:
#                     break
#                 else:
#                     text.append([var for var in value[i][1:]])
#
#     # get the text for token_nize
#     tokens = []
#     for i in range(len(text)):
#         for j in range(len(text[i])):
#             tokens.append(text[i][j])
#     # delete the blank value
#     tokens = [var for var in tokens if var]
#     tokens = set(tokens)
#
#     return tokens
#
# tokens = tokens_generate(key_para_dict)
# #
# #
# def token_dict(tokens):
#     '''
#     :param tokens: all the word tokens in the parameter value vector column
#     :return: token_encode_dict: the format is ['fawjeiajet';[32,45,65,..],...]
#     '''
#
#     # build the dict about different value
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(tokens)
#     encoded_texts = tokenizer.texts_to_sequences(tokens)
#     # build the dict with tokens --> encoded_texts
#     token_encode_dict = {}
#     for token, encoded_text in zip(tokens, encoded_texts):
#         token_encode_dict[token] = encoded_text
#
#     return token_encode_dict
#
# # the format is 'ate awte awet':[34,234,13]
# token_encode_dict = token_dict(tokens)
#
#
# def split_vectors(fd_id):
#     '''
#     :param fd_id: copied fd dataframe, in order to protect the original data
#     :return: fd_id: csv with parameter value vector splitted into various columns according to the max length of vector
#             list_name: the format is: value0, value1, value2, ....
#     '''
#     list_length = []
#     for var in fd_id['parameter value vector']:
#         list_length.append(len(var.split(',')))
#     # max(list_length) ---- 16
#     # list_length
#     list_name = []
#     for i in range(max(list_length)):
#         list_name.append('value' + str(i))
#     fd_id[list_name] = fd_id['parameter value vector'].str.split(",", expand=True, )
#     # [var for var in fd_id['value15'] if var]
#     # fd_id
#     for name in list_name:
#         for var in range(len(fd_id[name])):
#             # we should use fd_id[x] to rewrite value in
#             if fd_id[name][var] != None:
#                 fd_id[name][var] = re.sub("[\[|\]|']|\s+|\.|\-", '', fd_id[name][var])
#     fd_id.to_csv('../data/System_logs/log_value_vector_value.csv', index=False)
#
#     return fd_id, list_name
# # split the parameter value vector into different columns
#
# fd_id , list_name = split_vectors(fd_id)
#
#
# def map_vectors(fd_id, list_name):
#     '''
#     :param fd_id: csv with parameter value vector splitted into various columns according to the max length of vector
#     :param list_name: the format is: value0, value1, value2, ....
#     :return: fd_value: csv with textual values in parameter value vector replaced by numerical values
#     '''
#     # fd_value = pd.read_csv('../data/System_logs/log_value_vector_value.csv',delimiter=',', skipinitialspace=True)
#     fd_value = fd_id
#     for var in range(1,len(list_name)):
#         fd_value[list_name[var]] = fd_value[list_name[var]].map(token_encode_dict)
#     fd_value.to_csv('../data/System_logs/log_value_vector_value.csv',index = False)
#
#     return fd_value
#
# # fd_values = map_vectors(fd_id, list_name)
# fd_value = map_vectors(fd_id, list_name)

# ================== the part to integrate the columns into one column ==================
# ------ first step to execute --------
# fd_value['ColumnX'] = fd_value[fd_value.columns[3:19]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
# fd_value = fd_value.drop(['parameter value vector'], axis=1)
# fd_value = fd_value.drop(list_name, axis=1)
# # fd_value
# # -------- second step to execute --------------
# fd_value['parameter value vector'] = fd_value['ColumnX']
# fd_value = fd_value.drop(['ColumnX'], axis=1)
# # -------- third step to execute ---------------
# got the integrated column with numerical data in the parameter value vector
# fd_value.to_csv('../data/System_logs/log_value_vector_value.csv',index = False)
#
# ======================= step2 ==========================
# get all the parameter value vector for every unique key log
# fd_values = pd.read_csv('../data/System_logs/log_value_vector_value.csv')
#
# def log_vectors(fd_values):
#     '''
#     :param fd_values: csv with numerical values in a single parameter value vector column
#     :return: key_num_para_dict: the format is {Exx:[numerical parameter 1],[numerical parameter 2],...}
#     '''
#     uni_log_key_id = list(set(fd_values['log key']))
#
#     parameters = []
#
#     key_num_para_dict = {}
#     for i in range(len(uni_log_key_id)):
#         # get all the parameters with the same eventID
#         parameters = fd_values[fd_values['log key'] == uni_log_key_id[i]]['parameter value vector']
#         # the format will be E88: [xx,xx,xx],[xf,er,et],[re,tet,tet],...
#         key_num_para_dict[uni_log_key_id[i]] = parameters.values[:]
#     # plt.hist(key_para_dict['E88'], bins=15)
#     # plt.ylable('Counter')
#     # plt.xlabel('Variables')
#     # plt.title('Single Log Key Vectors')
#     print("the num para dict is:",key_num_para_dict)
#
#     df_dict_num_para = pd.DataFrame(dict([(k, Series(v)) for k, v in key_num_para_dict.items()]))
#     df_dict_num_para.to_csv('../data/System_logs/key_num_para_dict.csv',index= False, header=key_num_para_dict.keys())
#     return key_num_para_dict
#
# key_num_para_dict = log_vectors(fd_values)
#
# ## we use functions above to generate the numerical value vector in log_value_vector_value.csv
#
#
# # define the module to transform str into matrix
# # the string is like: '10635,[21, 85, 16, 18],[21, 85, 16, 18, 307, 308, 1],[356],[424],[207]'
# def str_array(dict, eventID):
#     '''
#     :param dict: the format is key:[numerical parameter 1],[numerical parameter 2],..
#     :param eventID: Exx
#     :return: saved matrix for unique event
#     '''
#     lists = dict[eventID]
#     list_string = []
#     pattern = '\d+'
#     numx = len(lists)
#     numy = len(re.findall(pattern, lists[0]))
#     list_array = np.empty(shape=[0, numy])
#     for string in lists:
#         # matching all digits
#         list_string = re.findall(pattern, string)
#         # transform the str into int
#         list_string = [int(var) for var in list_string]
#         list_string = np.array(list_string)
#         # concatenate multi lines
#         list_array = np.append(list_array, [list_string], axis = 0)
#     np.save('Event_npy/'+eventID+'.npy',list_array)
#
# # create all the matrixes for all the eventIDs
# for key in key_num_para_dict.keys():
#     matrix = str_array(key_num_para_dict, key)
#
#
# ===================== step 3 =======================
# ========================= Anomaly Detection Part =======================
#

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

    return model


if __name__ == "__main__":
    # use loop to input the name of matrix and get corresponding dataframe
    filenames = []
    root_dir = 'Event_npy/'
    # r=root, d = directories, f=files
    for r, d, f in os.walk(root_dir):
        for file in f:
            if '.npy' in file:
                filenames.append(os.path.join(r, file))
    rmses = []
    rmses_dict = {}

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
                train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.4, random_state=0)

            elif matrix.shape[0]>=4:
                n_steps = 1
                X, Y = training_data_generate(matrix, n_steps)
                # test_x and
                train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.5, random_state=0)
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
            # plt.hist(means, bins=yhat.shape[1])
            # x_list = []
            # for i in range(yhat.shape[1]):
            #     x_list.append(i)
            # plt.bar(x_list, means)
            # plt.ylabel("Errors Values")
            # plt.title('Errors Distribution')
            # plt.show()

            # use the mean square error to compare the difference between predicted y and validation y
            # the error follows the Gaussian distribution ---- normal, otherwise abnormal
            rmses.append(rmse)
            # save the result
            rmses_dict[file] = rmse
            # save the results to files
            joblib.dump(rmses, file+'_rmses.pkl')
            joblib.dump(rmses_dict, file+'_rmses_dict.pkl')
        # create the x axis labels for plot
        x_list = []
        for i in range(len(rmses)):
            x_list.append(i)
        plt.bar(x_list, rmses)
        plt.ylabel("Errors Values")
        plt.title('Errors Distribution')
        plt.show()
        print("the rmses_dict is {}".format(rmses_dict))
        print("the mean of rmses is: {}".format(np.mean(rmses)))

    '''
    normalization first for the whole dataset
    the mean of rmses is: 0.38624247585884824

    normalization after the split: normalize the train_x and test_x
    the mean of rmses is: 3320.1835964733777
    '''

# import joblib

# print(joblib.load('Event_npy/rmses.pkl'))
# print(joblib.load('Event_npy/rmses_dict.pkl'))
