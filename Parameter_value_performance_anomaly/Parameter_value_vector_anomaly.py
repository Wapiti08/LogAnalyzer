import sys
sys.path.append('../')

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

# load the module to generate the Dataframe for vector detection
import log_value_vector
# load the visualized part (if needed)
from Parameter_value_performance_anomaly import visualized_value_vector
# load the module to transform the value vector to numerical data --- step1
from Parameter_value_performance_anomaly import transform_numerical_data
# load the module to generate the matrix for every log key sequence --- step2
from Parameter_value_performance_anomaly import gen_log_key_matrix
# load the module to analyse the matrix for anomaly detection --- step3
from Parameter_value_performance_anomaly import matrix_analyse_report_anomaly


if __name__ == "__main__":

# ================== part1 to generate the dataframe for parameter detection =================
    # load the structured log csv
    structured_log_filename = None
    while True:
        try:
            print("Please input the right path with this kind of example: Dataset/xxx/Integrated_structured_log.csv")
            structured_log_filename = input("Please input the structured filename with full path:")
        except Exception as e:
            print(e)
        else:
            break

    fd_linux = pd.read_csv(structured_log_filename)
    fd_linux = fd_linux.copy()
    # create the first column in dataframe with 'log message'..
    log_messages = log_value_vector.log_messages_create(fd_linux)
    log_keys = fd_linux['EventTemplate']

    # append the timestamps to the log message
    month_list, day_list, time_list = [], [], []
    for i in range(len(fd_linux['Time'])):
        time_list.append(fd_linux['Time'][i].split(':'))
    for j in range(len(fd_linux['Date'])):
        day_list.append(fd_linux['Date'][j])
    # initialize the month_number
    month_number = 0
    # generate the first value (time_gap) in
    for k in range(len(fd_linux['Month'])):
        # print("we are transferring the month:",fd_linux['Month'][k])
        month_number = log_value_vector.month_string_to_number(fd_linux['Month'][k])
        month_list.append(month_number)
    # call the generated month, day, time list to get the seconds
    seconds_list = log_value_vector.trans_seconds(month_list, day_list, time_list)
    # input the seconds_list into the time_gap functions to get the series of time difference
    times = log_value_vector.time_gap(seconds_list)
    # process special case in log message to generate right values
    for i in range(len(fd_linux['Content'])):
        fd_linux['Content'][i] = re.sub('(\(\))', '(0)', fd_linux['Content'][i])
    fd_linux.to_csv(structured_log_filename, index=False)

    strings, strings1 = fd_linux['Content'], fd_linux['EventTemplate']
    # generate the later values in parameter value vector
    batch_variables = log_value_vector.show_diff(strings=strings, strings1=strings1)
    # define the pattern of system log, we can change it depends on the format of system log
    system_log_pattern = '(\S+)(\s+)(\w+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(.*)'

    while True:
        # generate the final dataframe for vector anomaly detection
        try:
            print("Please input the right path with this kind of example: Dataset/xxx/log_value_vector_mali.csv")
            log_value_vector_filename = input("Please input the file path you want to save for generated parameter dataframe:")
        except Exception as e:
            print(e)
        else:
            break

    log_value_vector_csv_fd = log_value_vector.parameter_vector(log_messages, log_keys, times, batch_variables, log_value_vector_filename)

# ================== part2 to transform the textual value to numerical data ================
    key_para_dict_filename = '../Dataset/Linux/Malicious_Separate_Structured_Logs/key_para_dict.csv'
    key_para_dict, fd_id = transform_numerical_data.vocabulary_generate(log_value_vector_csv_fd, key_para_dict_filename)

    # module to process the exception in template computation
    tokens = transform_numerical_data.tokens_generate(key_para_dict)
    # save the tokens_dict_filename
    tokens_dict_filename = '../Dataset/Linux/Malicious_Separate_Structured_Logs/tokens_dict.pkl'
    tokens_encode_dict = transform_numerical_data.token_dict(tokens, tokens_dict_filename)

    # split the parameter value vector into different columns
    # fd_id is the copied csv and list_name is like value x
    fd_id, list_name = transform_numerical_data.split_vectors(fd_id, log_value_vector_filename)

    # replace the textual data to numerical data
    fd_value = transform_numerical_data.map_vectors(fd_id, list_name, log_value_vector_filename)

    # integrate the vector lines into one
    integrated_fd_value = transform_numerical_data.integrate_lines(fd_value)

    # delete repeated column in the csv
    transform_numerical_data.delete_repeated_line(integrated_fd_value, log_value_vector_filename)


# ================== part3 to generate the separate matrix for log key ===================
    # we have the parameter file --- log_value_vector_filename
    fd_parameter = pd.read_csv(log_value_vector_filename)
    fd_parameter = fd_parameter.copy()
    key_num_para_dict = gen_log_key_matrix.log_vectors(fd_parameter)

    # create all the matrixes for all the eventIDs
    for key in key_num_para_dict.keys():
        print("the key is:", key)
        gen_log_key_matrix.str_array(key_num_para_dict, key)
# ================== part4 to analyse the matrix ===================

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
        if os.path.isfile(file + '_rmses.pkl'):
            rmses = joblib.load(file + '_rmses.pkl')

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

            elif matrix.shape[0] >= 4:
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
            joblib.dump(rmses, file + '_rmses.pkl')
            joblib.dump(rmses_dict, file + '_rmses_dict.pkl')

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
        file_number = re.findall('\d+', file)
        print("the file_number is:", file_number)
        plt.title(file_number[0] + ' ' + 'Errors Distribution')
        # plt.title(file + ' ' + 'Errors Distribution')
        plt.show()

        #  use normaltest to calculate the similarity between the errors and a Gaussian distribution
        #  suitable for examples above 8
        #  if len(rmses) >= 3:
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









